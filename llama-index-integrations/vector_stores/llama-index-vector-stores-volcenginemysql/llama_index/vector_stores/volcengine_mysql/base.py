"""
Volcengine RDS MySQL VectorStore integration for LlamaIndex.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import quote_plus

import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine
from llama_index.core.bridge.pydantic import PrivateAttr

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)


_logger = logging.getLogger(__name__)


@dataclass
class DBEmbeddingRow:
    """Internal helper struct representing a row fetched from the DB."""

    node_id: str
    text: str
    metadata: Dict[str, Any]
    similarity: float


class VolcengineMySQLVectorStore(BasePydanticVectorStore):
    """
    Volcengine RDS MySQL Vector Store.

    LlamaIndex vector store implementation backed by Volcengine RDS
    MySQL with native vector index support (``VECTOR(N)`` + HNSW ANN).

    Capabilities
    ~~~~~~~~~~~~
    - Vector column: ``VECTOR(embed_dim)``.
    - Vector index: ``VECTOR INDEX (embedding) USING HNSW`` or a vector
      index with ``SECONDARY_ENGINE_ATTRIBUTE`` specifying algorithm,
      ``M``, and distance metric, for example::

        SECONDARY_ENGINE_ATTRIBUTE='{"algorithm": "hnsw", "M": "16", "distance": "l2"}'

    - Distance functions:
      - ``L2_DISTANCE(embedding, TO_VECTOR('[...]'))``
      - ``COSINE_DISTANCE(embedding, TO_VECTOR('[...]'))``
    - Server parameters (depending on configuration):
      - ``loose_vector_index_enabled``
      - ``loose_hnsw_ef_search`` and other HNSW-related options.

    Differences from :class:`MariaDBVectorStore`
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Uses MySQL ``VECTOR`` columns and ``TO_VECTOR``/``L2_DISTANCE``
      functions instead of MariaDB's ``VECTOR(...)`` together with
      ``VEC_FromText``/``VEC_DISTANCE_COSINE``.
    - Uses ``JSON_EXTRACT`` / ``JSON_UNQUOTE`` to filter on the metadata
      JSON column.
    - Optionally uses ``loose_hnsw_ef_search`` to control ANN search
      breadth.
    """

    # LlamaIndex protocol flags
    stores_text: bool = True
    flat_metadata: bool = False

    # Pydantic model fields ( persisted configuration )
    connection_string: str
    connection_args: Dict[str, Any]
    table_name: str
    database: str
    embed_dim: int
    ann_index_algorithm: str
    ann_index_distance: str
    ann_m: int
    ef_search: int
    perform_setup: bool
    debug: bool

    # Runtime-only attributes
    _engine: Any = PrivateAttr()
    _aengine: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        connection_string: Union[str, sqlalchemy.engine.URL],
        connection_args: Optional[Dict[str, Any]] = None,
        table_name: str = "llamaindex",
        database: Optional[str] = None,
        embed_dim: int = 1536,
        ann_index_algorithm: str = "hnsw",
        ann_index_distance: str = "l2",
        ann_m: int = 16,
        ef_search: int = 20,
        perform_setup: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Constructor.

        Args:
            connection_string: SQLAlchemy/MySQL connection string, for
                example ``mysql+pymysql://user:pwd@host:3306/database``.
            connection_args: Extra connection arguments passed to
                SQLAlchemy. For Volcengine RDS MySQL this typically
                includes SSL options and read timeouts.
            table_name: Name of the table used to store vectors. Defaults
                to ``"llamaindex"``.
            database: Name of the database/schema (for bookkeeping only;
                the actual target is taken from the connection string).
            embed_dim: Embedding dimension. Must match the upstream
                embedding model dimension.
            ann_index_algorithm: Vector index algorithm. RDS MySQL
                currently supports ``"hnsw"``.
            ann_index_distance: Distance metric, ``"l2"`` or
                ``"cosine"``.
            ann_m: HNSW parameter ``M`` (maximum number of neighbors per
                node). Affects recall and performance.
            ef_search: HNSW ``ef_search`` parameter controlling search
                breadth at query time.
            perform_setup: If ``True``, perform basic capability checks
                and create the table/index on initialization.
            debug: If ``True``, enable SQLAlchemy SQL logging.

        """
        super().__init__(
            connection_string=str(connection_string),
            connection_args=connection_args
            or {
                "ssl": {"ssl_mode": "PREFERRED"},
                "read_timeout": 30,
            },
            table_name=table_name,
            database=database or "",
            embed_dim=embed_dim,
            ann_index_algorithm=ann_index_algorithm.lower(),
            ann_index_distance=ann_index_distance.lower(),
            ann_m=ann_m,
            ef_search=ef_search,
            perform_setup=perform_setup,
            debug=debug,
        )

        # Private attrs
        self._engine = None
        self._aengine = None
        self._is_initialized = False

    # ------------------------------------------------------------------
    # LlamaIndex base metadata
    # ------------------------------------------------------------------

    @classmethod
    def class_name(cls) -> str:
        """Return the vector store type name used by LlamaIndex."""
        return "VolcengineMySQLVectorStore"

    @property
    def client(self) -> Any:  # type: ignore[override]
        """Return the underlying SQLAlchemy engine (if initialized)."""
        if not self._is_initialized:
            return None
        return self._engine

    @property
    def aclient(self) -> Any:  # type: ignore[override]
        """Return the underlying Async SQLAlchemy engine (if initialized)."""
        if not self._is_initialized:
            return None
        return self._aengine

    def close(self) -> None:
        """Dispose the underlying SQLAlchemy engine."""
        if not self._is_initialized:
            return

        assert self._engine is not None
        self._engine.dispose()
        self._is_initialized = False

    async def aclose(self) -> None:
        """Dispose the underlying Async SQLAlchemy engine."""
        if self._aengine is not None:
            await self._aengine.dispose()
            self._aengine = None

    # ------------------------------------------------------------------
    # Factory construction
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "llamaindex",
        connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        embed_dim: int = 1536,
        ann_index_algorithm: str = "hnsw",
        ann_index_distance: str = "l2",
        ann_m: int = 16,
        ef_search: int = 20,
        perform_setup: bool = True,
        debug: bool = False,
    ) -> "VolcengineMySQLVectorStore":
        """
        Construct a vector store from basic connection parameters.

        Args:
            host: Hostname of the Volcengine RDS MySQL instance.
            port: Port of the MySQL instance (typically 3306).
            database: Database/schema name.
            user: Database username.
            password: Database password.
            table_name: Name of the table used to store vectors.
            connection_string: Optional full SQLAlchemy connection string.
                If provided, it takes precedence over ``host``/``user``/
                ``password``/``database``.
            connection_args: Optional dict of extra SQLAlchemy connection
                arguments.
            embed_dim: Embedding dimension.
            ann_index_algorithm: Vector index algorithm, typically
                ``"hnsw"``.
            ann_index_distance: Distance metric, ``"l2"`` or
                ``"cosine"``.
            ann_m: HNSW ``M`` parameter.
            ef_search: HNSW ``ef_search`` parameter.
            perform_setup: Whether to create the table/index and validate
                configuration on initialization.
            debug: Whether to emit SQL debug logs.

        """
        if connection_string is None:
            if not all([host, port, database, user]):
                raise ValueError(
                    "host/port/database/user must all be provided, or pass a full connection_string instead."
                )
            password_safe = quote_plus(password or "")
            connection_string = (
                f"mysql+pymysql://{user}:{password_safe}@{host}:{port}/{database}"
            )

        return cls(
            connection_string=connection_string,
            connection_args=connection_args,
            table_name=table_name,
            database=database,
            embed_dim=embed_dim,
            ann_index_algorithm=ann_index_algorithm,
            ann_index_distance=ann_index_distance,
            ann_m=ann_m,
            ef_search=ef_search,
            perform_setup=perform_setup,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Internal initialization & DDL
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Create SQLAlchemy engine."""
        self._engine = sqlalchemy.create_engine(
            self.connection_string,
            connect_args=self.connection_args,
            echo=self.debug,
        )

    def _aconnect(self) -> None:
        """Create Async SQLAlchemy engine."""
        if self._aengine is not None:
            return

        # Prepare async connection string
        # We replace 'pymysql' with 'aiomysql' if present
        async_conn_str = self.connection_string.replace("pymysql", "aiomysql")

        # aiomysql does not support 'read_timeout' which is commonly used in pymysql
        # Filter out incompatible args
        filtered_args = {
            k: v for k, v in self.connection_args.items() if k != "read_timeout"
        }

        self._aengine = create_async_engine(
            async_conn_str,
            connect_args=filtered_args,
            echo=self.debug,
        )

    def _validate_server_capability(self) -> None:
        """
        Validate that the MySQL server supports Volcengine vector index.

        The current implementation performs only a basic check:

        - Run ``SHOW VARIABLES LIKE 'loose_vector_index_enabled'`` and
          verify that the value is ``ON``.
        - If the variable is missing or disabled, raise an error and ask
          the user to enable it in the RDS console or parameter template.

        This method can be extended to also inspect ``SELECT VERSION()``
        and enforce a minimum server version if needed.
        """
        assert self._engine is not None

        with self._engine.connect() as connection:
            # Check loose_vector_index_enabled
            result = connection.execute(
                sqlalchemy.text("SHOW VARIABLES LIKE :var"),
                {"var": "loose_vector_index_enabled"},
            )
            row = result.fetchone()
            if not row or str(row[1]).upper() != "ON":
                raise ValueError(
                    "Volcengine MySQL vector index is not enabled: please set loose_vector_index_enabled to ON."
                )

    def _create_table_if_not_exists(self) -> None:
        """
        Create table with a VECTOR column and HNSW vector index if needed.

        Example schema::

            CREATE TABLE IF NOT EXISTS `table_name` (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                node_id VARCHAR(255) NOT NULL,
                text LONGTEXT,
                metadata JSON,
                embedding VECTOR(1536) NOT NULL,
                INDEX idx_node_id (node_id),
                VECTOR INDEX idx_embedding (embedding)
                  SECONDARY_ENGINE_ATTRIBUTE='{"algorithm": "hnsw", "M": "16", "distance": "l2"}'
            ) ENGINE = InnoDB;

        Notes
        -----
        - Vector indexes can typically only be created on empty tables.
          It is therefore recommended to let this class create the table
          *before* any data is written.
        - If a user has already created the table without a vector
          index, this method will **not** attempt to run
          ``ALTER TABLE ... ADD VECTOR INDEX`` on existing data in order
          to avoid long locks or failures. In that case, please migrate
          the data manually or create the correct schema ahead of time.

        """
        assert self._engine is not None

        sec_attr = (
            "{"  # Build JSON string for SECONDARY_ENGINE_ATTRIBUTE
            f'"algorithm": "{self.ann_index_algorithm}", '
            f'"M": "{self.ann_m}", '
            f'"distance": "{self.ann_index_distance}"'
            "}"
        )

        create_stmt = f"""
        CREATE TABLE IF NOT EXISTS `{self.table_name}` (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            node_id VARCHAR(255) NOT NULL,
            text LONGTEXT,
            metadata JSON,
            embedding VECTOR({self.embed_dim}) NOT NULL,
            INDEX idx_node_id (node_id),
            VECTOR INDEX idx_embedding (embedding)
              SECONDARY_ENGINE_ATTRIBUTE='{sec_attr}'
        ) ENGINE = InnoDB
        """

        with self._engine.connect() as connection:
            connection.execute(sqlalchemy.text(create_stmt))
            connection.commit()

    def _initialize(self) -> None:
        """Ensure engine is created and table is ready."""
        if self._engine is None:
            self._connect()

        if self._is_initialized:
            return

        if self.perform_setup:
            self._validate_server_capability()
            self._create_table_if_not_exists()

        self._is_initialized = True

    async def _ainitialize(self) -> None:
        """Ensure async engine is created and table is ready."""
        if self._aengine is None:
            self._aconnect()

        if self._is_initialized:
            return

        if self.perform_setup:
            await self._avalidate_server_capability()
            await self._acreate_table_if_not_exists()

        self._is_initialized = True

    async def _avalidate_server_capability(self) -> None:
        """Async version of _validate_server_capability."""
        assert self._aengine is not None

        async with self._aengine.connect() as connection:
            result = await connection.execute(
                sqlalchemy.text("SHOW VARIABLES LIKE :var"),
                {"var": "loose_vector_index_enabled"},
            )
            row = result.fetchone()
            if not row or str(row[1]).upper() != "ON":
                raise ValueError(
                    "Volcengine MySQL vector index is not enabled: please set loose_vector_index_enabled to ON."
                )

    async def _acreate_table_if_not_exists(self) -> None:
        """Async version of _create_table_if_not_exists."""
        assert self._aengine is not None

        sec_attr = (
            "{"  # Build JSON string for SECONDARY_ENGINE_ATTRIBUTE
            f'"algorithm": "{self.ann_index_algorithm}", '
            f'"M": "{self.ann_m}", '
            f'"distance": "{self.ann_index_distance}"'
            "}"
        )

        create_stmt = f"""
        CREATE TABLE IF NOT EXISTS `{self.table_name}` (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            node_id VARCHAR(255) NOT NULL,
            text LONGTEXT,
            metadata JSON,
            embedding VECTOR({self.embed_dim}) NOT NULL,
            INDEX idx_node_id (node_id),
            VECTOR INDEX idx_embedding (embedding)
              SECONDARY_ENGINE_ATTRIBUTE='{sec_attr}'
        ) ENGINE = InnoDB
        """

        async with self._aengine.connect() as connection:
            await connection.execute(sqlalchemy.text(create_stmt))
            await connection.commit()

    # ------------------------------------------------------------------
    # Helpers for (de)serializing nodes and filters
    # ------------------------------------------------------------------

    def _node_to_table_row(self, node: BaseNode) -> Dict[str, Any]:
        """Convert a BaseNode into a plain row dict ready for insertion."""
        return {
            "node_id": node.node_id,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "embedding": node.get_embedding(),
            "metadata": node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            ),
        }

    def _to_mysql_operator(self, operator: FilterOperator) -> str:
        """Map LlamaIndex FilterOperator to SQL operator string."""
        if operator == FilterOperator.EQ:
            return "="
        if operator == FilterOperator.GT:
            return ">"
        if operator == FilterOperator.LT:
            return "<"
        if operator == FilterOperator.NE:
            return "!="
        if operator == FilterOperator.GTE:
            return ">="
        if operator == FilterOperator.LTE:
            return "<="
        if operator == FilterOperator.IN:
            return "IN"
        if operator == FilterOperator.NIN:
            return "NOT IN"

        _logger.warning("Unsupported operator: %s, fallback to '='", operator)
        return "="

    def _build_filter_clause(
        self,
        filter_: MetadataFilter,
        params: Dict[str, Any],
        param_counter: List[int],
    ) -> str:
        """
        Build a single metadata filter expression for the JSON column.

        Rules:
        - For string values use ``JSON_UNQUOTE(JSON_EXTRACT(...))`` in
          comparisons.
        - For numeric values compare the result of ``JSON_EXTRACT(...)``
          directly.
        - For ``IN``/``NIN`` operators build a ``(v1, v2, ...)`` value
          list.
        """
        key_expr = f"JSON_EXTRACT(metadata, '$.{filter_.key}')"
        value = filter_.value

        if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
            assert isinstance(value, list), (
                "The value for an IN/NIN filter must be a list"
            )
            param_keys: List[str] = []
            for v in value:
                param_name = f"filter_param_{param_counter[0]}"
                param_counter[0] += 1

                # For IN/NIN, we always compare as strings after JSON_UNQUOTE
                if isinstance(v, str):
                    params[param_name] = v
                else:
                    params[param_name] = str(v)

                param_keys.append(f":{param_name}")

            filter_value = f"({', '.join(param_keys)})"
            return f"JSON_UNQUOTE({key_expr}) {self._to_mysql_operator(filter_.operator)} {filter_value}"

        # Scalar comparison
        param_name = f"filter_param_{param_counter[0]}"
        param_counter[0] += 1
        params[param_name] = value

        if isinstance(value, str):
            expr = f"JSON_UNQUOTE({key_expr}) {self._to_mysql_operator(filter_.operator)} :{param_name}"
        else:
            # For numeric or other non-string values, compare the JSON_EXTRACT
            # result directly.
            expr = (
                f"{key_expr} {self._to_mysql_operator(filter_.operator)} :{param_name}"
            )

        return expr

    def _filters_to_where_clause(
        self,
        filters: MetadataFilters,
        params: Dict[str, Any],
        param_counter: List[int],
    ) -> str:
        """Convert MetadataFilters tree into a SQL WHERE clause (without 'WHERE')."""
        conditions_map = {
            FilterCondition.OR: "OR",
            FilterCondition.AND: "AND",
        }

        if filters.condition not in conditions_map:
            raise ValueError(
                f"Unsupported condition: {filters.condition}. "
                f"Must be one of {list(conditions_map.keys())}"
            )

        clauses: List[str] = []
        for f in filters.filters:
            if isinstance(f, MetadataFilter):
                clauses.append(self._build_filter_clause(f, params, param_counter))
            elif isinstance(f, MetadataFilters):
                sub = self._filters_to_where_clause(f, params, param_counter)
                if sub:
                    clauses.append(f"({sub})")
            else:
                raise ValueError(
                    "Unsupported filter type: {type(f)}. Must be one of "
                    f"MetadataFilter, MetadataFilters"
                )

        return f" {conditions_map[filters.condition]} ".join(clauses)

    def _db_rows_to_query_result(
        self, rows: List[DBEmbeddingRow]
    ) -> VectorStoreQueryResult:
        """Convert internal DB rows to LlamaIndex VectorStoreQueryResult."""
        nodes: List[BaseNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        for r in rows:
            metadata = r.metadata or {}

            # If the metadata contains the special fields used by
            # `metadata_dict_to_node`, reconstruct the original node.
            # Otherwise, fall back to a plain TextNode so that we can still
            # return meaningful results when only custom metadata is stored.
            if isinstance(metadata, dict) and metadata.get("_node_content") is not None:
                node = metadata_dict_to_node(metadata)
                node.set_content(str(r.text))
            else:
                node = TextNode(text=str(r.text), id_=r.node_id, metadata=metadata)
            nodes.append(node)
            ids.append(r.node_id)
            similarities.append(r.similarity)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    # ------------------------------------------------------------------
    # Public API: get_nodes / add / delete / query
    # ------------------------------------------------------------------

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:  # type: ignore[override]
        """
        Get nodes by ``node_ids``.

        Note:
            The current implementation only supports exact lookup by
            ``node_ids`` and ignores the ``filters`` argument.

        """
        self._initialize()

        if not node_ids:
            return []

        # Use bind parameters for the IN clause
        stmt_str = (
            f"SELECT text, metadata FROM `{self.table_name}` WHERE node_id IN :node_ids"
        )
        stmt = sqlalchemy.text(stmt_str).bindparams(
            sqlalchemy.bindparam("node_ids", expanding=True)
        )

        assert self._engine is not None
        with self._engine.connect() as connection:
            result = connection.execute(stmt, {"node_ids": node_ids})

        nodes: List[BaseNode] = []
        for item in result:
            raw_meta = item.metadata
            metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta

            if isinstance(metadata, dict) and metadata.get("_node_content") is not None:
                node = metadata_dict_to_node(metadata)
                node.set_content(str(item.text))
            else:
                node = TextNode(text=str(item.text), metadata=metadata or {})

            nodes.append(node)

        return nodes

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:  # type: ignore[override]
        """
        Add nodes with embeddings into the MySQL vector store.

        Expectations:
        - Each :class:`BaseNode` in ``nodes`` must already contain an
          ``embedding`` (normally computed by the index or embedding
          model upstream).
        - The embedding is serialized as a JSON array string and passed
          to ``TO_VECTOR(:embedding)`` when inserting into the
          ``VECTOR`` column.
        - Rows are inserted in batch using ``executemany`` semantics to
          reduce round trips.
        """
        self._initialize()

        if not nodes:
            return []

        ids: List[str] = []
        rows: List[Dict[str, Any]] = []

        for node in nodes:
            ids.append(node.node_id)
            item = self._node_to_table_row(node)
            rows.append(
                {
                    "node_id": item["node_id"],
                    "text": item["text"],
                    # TO_VECTOR expects a string like "[1.0,2.0,...]"
                    "embedding": json.dumps(item["embedding"]),
                    "metadata": json.dumps(item["metadata"]),
                }
            )

        insert_stmt = sqlalchemy.text(
            f"""
            INSERT INTO `{self.table_name}` (node_id, text, embedding, metadata)
            VALUES (:node_id, :text, TO_VECTOR(:embedding), :metadata)
            """
        )

        assert self._engine is not None
        with self._engine.connect() as connection:
            connection.execute(insert_stmt, rows)
            connection.commit()

        return ids

    async def async_add(  # type: ignore[override]
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add nodes with embeddings into the MySQL vector store asynchronously.
        """
        await self._ainitialize()

        if not nodes:
            return []

        ids: List[str] = []
        rows: List[Dict[str, Any]] = []

        for node in nodes:
            ids.append(node.node_id)
            item = self._node_to_table_row(node)
            rows.append(
                {
                    "node_id": item["node_id"],
                    "text": item["text"],
                    # TO_VECTOR expects a string like "[1.0,2.0,...]"
                    "embedding": json.dumps(item["embedding"]),
                    "metadata": json.dumps(item["metadata"]),
                }
            )

        insert_stmt = sqlalchemy.text(
            f"""
            INSERT INTO `{self.table_name}` (node_id, text, embedding, metadata)
            VALUES (:node_id, :text, TO_VECTOR(:embedding), :metadata)
            """
        )

        async with self._aengine.connect() as connection:
            await connection.execute(insert_stmt, rows)
            await connection.commit()

        return ids

    def delete(
        self,
        ref_doc_id: str,
        **delete_kwargs: Any,
    ) -> None:  # type: ignore[override]
        """Delete all nodes whose metadata.ref_doc_id equals the given value."""
        self._initialize()

        if not ref_doc_id:
            return

        stmt = sqlalchemy.text(
            f"""
            DELETE FROM `{self.table_name}`
            WHERE JSON_EXTRACT(metadata, '$.ref_doc_id') = :doc_id
            """
        )

        assert self._engine is not None
        with self._engine.connect() as connection:
            connection.execute(stmt, {"doc_id": ref_doc_id})
            connection.commit()

    async def adelete(  # type: ignore[override]
        self,
        ref_doc_id: str,
        **delete_kwargs: Any,
    ) -> None:
        """Async wrapper around :meth:`delete`."""
        await self._ainitialize()

        if not ref_doc_id:
            return

        stmt = sqlalchemy.text(
            f"""
            DELETE FROM `{self.table_name}`
            WHERE JSON_EXTRACT(metadata, '$.ref_doc_id') = :doc_id
            """
        )

        async with self._aengine.connect() as connection:
            await connection.execute(stmt, {"doc_id": ref_doc_id})
            await connection.commit()

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:  # type: ignore[override]
        """
        Delete nodes by ``node_ids``.

        Note:
            The current implementation only supports deletion by
            ``node_ids`` and ignores ``filters``.

        """
        self._initialize()

        if not node_ids:
            return

        stmt_str = f"DELETE FROM `{self.table_name}` WHERE node_id IN :node_ids"
        stmt = sqlalchemy.text(stmt_str).bindparams(
            sqlalchemy.bindparam("node_ids", expanding=True)
        )

        assert self._engine is not None
        with self._engine.connect() as connection:
            connection.execute(stmt, {"node_ids": node_ids})
            connection.commit()

    async def adelete_nodes(  # type: ignore[override]
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Async wrapper around :meth:`delete_nodes`."""
        await self._ainitialize()

        if not node_ids:
            return

        stmt_str = f"DELETE FROM `{self.table_name}` WHERE node_id IN :node_ids"
        stmt = sqlalchemy.text(stmt_str).bindparams(
            sqlalchemy.bindparam("node_ids", expanding=True)
        )

        async with self._aengine.connect() as connection:
            await connection.execute(stmt, {"node_ids": node_ids})
            await connection.commit()

    def count(self) -> int:
        """Return total number of rows in the table."""
        self._initialize()

        stmt = sqlalchemy.text(f"SELECT COUNT(*) FROM `{self.table_name}`")

        assert self._engine is not None
        with self._engine.connect() as connection:
            result = connection.execute(stmt)
            value = result.scalar()

        return int(value or 0)

    def drop(self) -> None:
        """Drop the underlying table and dispose the engine."""
        self._initialize()

        stmt = sqlalchemy.text(f"DROP TABLE IF EXISTS `{self.table_name}`")

        assert self._engine is not None
        with self._engine.connect() as connection:
            connection.execute(stmt)
            connection.commit()

        self.close()

    def clear(self) -> None:  # type: ignore[override]
        """Delete all rows from the table (keep schema & indexes)."""
        self._initialize()

        stmt = sqlalchemy.text(f"DELETE FROM `{self.table_name}`")

        assert self._engine is not None
        with self._engine.connect() as connection:
            connection.execute(stmt)
            connection.commit()

    async def aclear(self) -> None:  # type: ignore[override]
        """Async wrapper around :meth:`clear`."""
        await self._ainitialize()

        stmt = sqlalchemy.text(f"DELETE FROM `{self.table_name}`")

        async with self._aengine.connect() as connection:
            await connection.execute(stmt)
            await connection.commit()

    def _build_distance_expression(self) -> str:
        """
        Return the SQL distance expression template used in ORDER BY.

        The returned string uses a named bind parameter ``:query_embedding``
        (serialized JSON array string) and the ``embedding`` column.
        """
        if self.ann_index_distance == "cosine":
            func_name = "COSINE_DISTANCE"
        else:
            # Default to L2 distance
            func_name = "L2_DISTANCE"

        return f"{func_name}(embedding, TO_VECTOR(:query_embedding))"

    def query(  # type: ignore[override]
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Execute a vector similarity search on Volcengine RDS MySQL.

        - Only ``VectorStoreQueryMode.DEFAULT`` is supported.
        - The database-side vector index and distance functions are used
          to perform ANN/KNN search.
        - :class:`MetadataFilters` are translated into a ``WHERE``
          clause over the JSON ``metadata`` column.
        - Returned similarities are computed as ``1 / (1 + distance)``.
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"Query mode {query.mode} not available.")

        if query.query_embedding is None:
            raise ValueError(
                "VolcengineMySQLVectorStore only supports embedding-based queries; query_embedding must be provided"
            )

        self._initialize()

        distance_expr = self._build_distance_expression()

        base_stmt = f"""
        SELECT
            node_id,
            text,
            metadata,
            {distance_expr} AS distance
        FROM `{self.table_name}`
        """

        # Metadata filters
        params = {
            "query_embedding": json.dumps(query.query_embedding),
            "limit": int(query.similarity_top_k),
        }

        if query.filters is not None:
            param_counter = [0]
            where_clause = self._filters_to_where_clause(
                query.filters, params, param_counter
            )
            if where_clause:
                base_stmt += f" WHERE {where_clause}"

        base_stmt += " ORDER BY distance LIMIT :limit"

        rows: List[DBEmbeddingRow] = []

        assert self._engine is not None
        with self._engine.connect() as connection:
            # Optionally set ef_search, which affects recall and latency
            if self.ef_search:
                try:
                    connection.execute(
                        sqlalchemy.text(
                            "SET SESSION loose_hnsw_ef_search = :ef_search"
                        ),
                        {"ef_search": int(self.ef_search)},
                    )
                except Exception:  # pragma: no cover - tolerate cases where the parameter does not exist
                    _logger.warning(
                        "Failed to set loose_hnsw_ef_search, continue without it.",
                        exc_info=True,
                    )

            result = connection.execute(sqlalchemy.text(base_stmt), params)

            for item in result:
                raw_meta = item.metadata
                metadata = (
                    json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                )
                distance = float(item.distance) if item.distance is not None else 0.0
                similarity = 1.0 / (1.0 + distance)

                rows.append(
                    DBEmbeddingRow(
                        node_id=item.node_id,
                        text=item.text,
                        metadata=metadata,
                        similarity=similarity,
                    )
                )

        return self._db_rows_to_query_result(rows)

    async def aquery(  # type: ignore[override]
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Async wrapper around :meth:`query`."""
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"Query mode {query.mode} not available.")

        if query.query_embedding is None:
            raise ValueError(
                "VolcengineMySQLVectorStore only supports embedding-based queries; query_embedding must be provided"
            )

        await self._ainitialize()

        distance_expr = self._build_distance_expression()

        base_stmt = f"""
        SELECT
            node_id,
            text,
            metadata,
            {distance_expr} AS distance
        FROM `{self.table_name}`
        """

        # Metadata filters
        params = {
            "query_embedding": json.dumps(query.query_embedding),
            "limit": int(query.similarity_top_k),
        }

        if query.filters is not None:
            param_counter = [0]
            where_clause = self._filters_to_where_clause(
                query.filters, params, param_counter
            )
            if where_clause:
                base_stmt += f" WHERE {where_clause}"

        base_stmt += " ORDER BY distance LIMIT :limit"

        rows: List[DBEmbeddingRow] = []

        async with self._aengine.connect() as connection:
            # Optionally set ef_search, which affects recall and latency
            if self.ef_search:
                try:
                    await connection.execute(
                        sqlalchemy.text(
                            "SET SESSION loose_hnsw_ef_search = :ef_search"
                        ),
                        {"ef_search": int(self.ef_search)},
                    )
                except Exception:
                    _logger.warning(
                        "Failed to set loose_hnsw_ef_search, continue without it.",
                        exc_info=True,
                    )

            result = await connection.execute(sqlalchemy.text(base_stmt), params)

            for item in result:
                raw_meta = item.metadata
                metadata = (
                    json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                )
                distance = float(item.distance) if item.distance is not None else 0.0
                similarity = 1.0 / (1.0 + distance)

                rows.append(
                    DBEmbeddingRow(
                        node_id=item.node_id,
                        text=item.text,
                        metadata=metadata,
                        similarity=similarity,
                    )
                )

        return self._db_rows_to_query_result(rows)
