"""MariaDB Vector Store."""

import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Union
from urllib.parse import quote_plus

import sqlalchemy
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
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


class DBEmbeddingRow(NamedTuple):
    node_id: str
    text: str
    metadata: dict
    similarity: float


_logger = logging.getLogger(__name__)


class MariaDBVectorStore(BasePydanticVectorStore):
    """MariaDB Vector Store.

    Examples:
        `pip install llama-index-vector-stores-mariadb`

        ```python
        from llama_index.vector_stores.mariadb import MariaDBVectorStore

        # Create MariaDBVectorStore instance
        vector_store = MariaDBVectorStore.from_params(
            host="localhost",
            port=3306,
            user="llamaindex",
            password="password",
            database="vectordb",
            table_name="llama_index_vectorstore",
            default_m=6,
            ef_search=20,
            embed_dim=1536  # OpenAI embedding dimension
        )
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = False

    connection_string: str
    connection_args: Dict[str, Any]
    table_name: str
    schema_name: str
    embed_dim: int
    default_m: int
    ef_search: int
    perform_setup: bool
    debug: bool

    _engine: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        connection_string: Union[str, sqlalchemy.engine.URL],
        connection_args: Dict[str, Any],
        table_name: str,
        schema_name: str,
        embed_dim: int = 1536,
        default_m: int = 6,
        ef_search: int = 20,
        perform_setup: bool = True,
        debug: bool = False,
    ) -> None:
        """Constructor.

        Args:
            connection_string (Union[str, sqlalchemy.engine.URL]): Connection string for the MariaDB server.
            connection_args (Dict[str, Any]): A dictionary of connection options.
            table_name (str): Table name.
            schema_name (str): Schema name.
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            default_m (int, optional): Default M value for the vector index. Defaults to 6.
            ef_search (int, optional): EF search value for the vector index. Defaults to 20.
            perform_setup (bool, optional): If DB should be set up. Defaults to True.
            debug (bool, optional): Debug mode. Defaults to False.
        """
        super().__init__(
            connection_string=connection_string,
            connection_args=connection_args,
            table_name=table_name,
            schema_name=schema_name,
            embed_dim=embed_dim,
            default_m=default_m,
            ef_search=ef_search,
            perform_setup=perform_setup,
            debug=debug,
        )

        self._initialize()

    def close(self) -> None:
        if not self._is_initialized:
            return

        self._engine.dispose()
        self._is_initialized = False

    @classmethod
    def class_name(cls) -> str:
        return "MariaDBVectorStore"

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "llamaindex",
        schema_name: str = "public",
        connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        embed_dim: int = 1536,
        default_m: int = 6,
        ef_search: int = 20,
        perform_setup: bool = True,
        debug: bool = False,
    ) -> "MariaDBVectorStore":
        """Construct from params.

        Args:
            host (Optional[str], optional): Host of MariaDB connection. Defaults to None.
            port (Optional[str], optional): Port of MariaDB connection. Defaults to None.
            database (Optional[str], optional): MariaDB DB name. Defaults to None.
            user (Optional[str], optional): MariaDB username. Defaults to None.
            password (Optional[str], optional): MariaDB password. Defaults to None.
            table_name (str): Table name. Defaults to "llamaindex".
            schema_name (str): Schema name. Defaults to "public".
            connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to MariaDB DB.
            connection_args (Dict[str, Any], optional): A dictionary of connection options.
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            default_m (int, optional): Default M value for the vector index. Defaults to 6.
            ef_search (int, optional): EF search value for the vector index. Defaults to 20.
            perform_setup (bool, optional): If DB should be set up. Defaults to True.
            debug (bool, optional): Debug mode. Defaults to False.

        Returns:
            MariaDBVectorStore: Instance of MariaDBVectorStore constructed from params.
        """
        conn_str = (
            connection_string
            or f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{database}"
        )
        conn_args = connection_args or {
            "ssl": {"ssl_mode": "PREFERRED"},
            "read_timeout": 30,
        }

        return cls(
            connection_string=conn_str,
            connection_args=conn_args,
            table_name=table_name,
            schema_name=schema_name,
            embed_dim=embed_dim,
            default_m=default_m,
            ef_search=ef_search,
            perform_setup=perform_setup,
            debug=debug,
        )

    @property
    def client(self) -> Any:
        if not self._is_initialized:
            return None
        return self._engine

    def _connect(self) -> Any:
        self._engine = sqlalchemy.create_engine(
            self.connection_string, connect_args=self.connection_args, echo=self.debug
        )

    def _validate_server_version(self) -> None:
        """Validate that the MariaDB server version is supported."""
        with self._engine.connect() as connection:
            result = connection.execute(sqlalchemy.text("SELECT VERSION()"))
            version = result.fetchone()[0]

            if not _meets_min_server_version(version, "11.7.1"):
                raise ValueError(
                    f"MariaDB version 11.7.1 or later is required, found version: {version}."
                )

    def _create_table_if_not_exists(self) -> None:
        with self._engine.connect() as connection:
            # Note that we define the vector index with DISTANCE=cosine, because we use VEC_DISTANCE_COSINE.
            # This is because searches using a different distance function do not use the vector index.
            # Reference: https://mariadb.com/kb/en/create-table-with-vectors/
            stmt = f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                id SERIAL PRIMARY KEY,
                node_id VARCHAR(255) NOT NULL,
                text TEXT,
                metadata JSON,
                embedding VECTOR({self.embed_dim}) NOT NULL,
                INDEX (`node_id`),
                VECTOR INDEX (embedding) M={self.default_m} DISTANCE=cosine
            )
            """
            connection.execute(sqlalchemy.text(stmt))

            connection.commit()

    def _initialize(self) -> None:
        if not self._is_initialized:
            self._connect()
            if self.perform_setup:
                self._validate_server_version()
                self._create_table_if_not_exists()
            self._is_initialized = True

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes from vector store."""
        self._initialize()

        stmt = f"""SELECT text, metadata FROM `{self.table_name}` WHERE node_id IN :node_ids"""

        with self._engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(stmt), {"node_ids": node_ids})

        nodes: List[BaseNode] = []
        for item in result:
            node = metadata_dict_to_node(json.loads(item.metadata))
            node.set_content(str(item.text))
            nodes.append(node)

        return nodes

    def _node_to_table_row(self, node: BaseNode) -> Dict[str, Any]:
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

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        self._initialize()

        ids = []
        with self._engine.connect() as connection:
            for node in nodes:
                ids.append(node.node_id)
                item = self._node_to_table_row(node)
                stmt = sqlalchemy.text(
                    f"""
                INSERT INTO `{self.table_name}` (node_id, text, embedding, metadata)
                VALUES (
                    :node_id,
                    :text,
                    VEC_FromText(:embedding),
                    :metadata
                )
                """
                )
                connection.execute(
                    stmt,
                    {
                        "node_id": item["node_id"],
                        "text": item["text"],
                        "embedding": json.dumps(item["embedding"]),
                        "metadata": json.dumps(item["metadata"]),
                    },
                )

            connection.commit()

        return ids

    def _to_mariadb_operator(self, operator: FilterOperator) -> str:
        if operator == FilterOperator.EQ:
            return "="
        elif operator == FilterOperator.GT:
            return ">"
        elif operator == FilterOperator.LT:
            return "<"
        elif operator == FilterOperator.NE:
            return "!="
        elif operator == FilterOperator.GTE:
            return ">="
        elif operator == FilterOperator.LTE:
            return "<="
        elif operator == FilterOperator.IN:
            return "IN"
        elif operator == FilterOperator.NIN:
            return "NOT IN"
        else:
            _logger.warning("Unsupported operator: %s, fallback to '='", operator)
            return "="

    def _build_filter_clause(self, filter_: MetadataFilter) -> str:
        filter_value = filter_.value
        if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
            values = []
            for v in filter_.value:
                if isinstance(v, str):
                    value = f"'{v}'"

                values.append(value)
            filter_value = ", ".join(values)
            filter_value = f"({filter_value})"
        elif isinstance(filter_.value, str):
            filter_value = f"'{filter_.value}'"

        return f"JSON_VALUE(metadata, '$.{filter_.key}') {self._to_mariadb_operator(filter_.operator)} {filter_value}"

    def _filters_to_where_clause(self, filters: MetadataFilters) -> str:
        conditions = {
            FilterCondition.OR: "OR",
            FilterCondition.AND: "AND",
        }
        if filters.condition not in conditions:
            raise ValueError(
                f"Unsupported condition: {filters.condition}. "
                f"Must be one of {list(conditions.keys())}"
            )

        clauses: List[str] = []
        for filter_ in filters.filters:
            if isinstance(filter_, MetadataFilter):
                clauses.append(self._build_filter_clause(filter_))
                continue

            if isinstance(filter_, MetadataFilters):
                subfilters = self._filters_to_where_clause(filter_)
                if subfilters:
                    clauses.append(f"({subfilters})")
                continue

            raise ValueError(
                f"Unsupported filter type: {type(filter_)}. Must be one of {MetadataFilter}, {MetadataFilters}"
            )
        return f" {conditions[filters.condition]} ".join(clauses)

    def _db_rows_to_query_result(
        self, rows: List[DBEmbeddingRow]
    ) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []
        for db_embedding_row in rows:
            node = metadata_dict_to_node(db_embedding_row.metadata)
            node.set_content(str(db_embedding_row.text))

            similarities.append(db_embedding_row.similarity)
            ids.append(db_embedding_row.node_id)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"Query mode {query.mode} not available.")

        self._initialize()

        stmt = f"""
        SET STATEMENT mhnsw_ef_search={self.ef_search} FOR
        SELECT
            node_id,
            text,
            embedding,
            metadata,
            VEC_DISTANCE_COSINE(embedding, VEC_FromText('{query.query_embedding}')) AS distance
        FROM `{self.table_name}`"""

        if query.filters:
            stmt += f"""
        WHERE {self._filters_to_where_clause(query.filters)}"""

        stmt += f"""
        ORDER BY distance
        LIMIT {query.similarity_top_k}
        """

        with self._engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(stmt))

        results = []
        for item in result:
            results.append(
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=json.loads(item.metadata),
                    similarity=(1 - item.distance) if item.distance is not None else 0,
                )
            )

        return self._db_rows_to_query_result(results)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._initialize()

        with self._engine.connect() as connection:
            # Should we create an index on ref_doc_id?
            stmt = f"""DELETE FROM `{self.table_name}` WHERE JSON_EXTRACT(metadata, '$.ref_doc_id') = :doc_id"""
            connection.execute(sqlalchemy.text(stmt), {"doc_id": ref_doc_id})

            connection.commit()

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        self._initialize()

        with self._engine.connect() as connection:
            stmt = f"""DELETE FROM `{self.table_name}` WHERE node_id IN :node_ids"""
            connection.execute(sqlalchemy.text(stmt), {"node_ids": node_ids})

            connection.commit()

    def count(self) -> int:
        self._initialize()

        with self._engine.connect() as connection:
            stmt = f"""SELECT COUNT(*) FROM `{self.table_name}`"""
            result = connection.execute(sqlalchemy.text(stmt))

        return result.scalar() or 0

    def drop(self) -> None:
        self._initialize()

        with self._engine.connect() as connection:
            stmt = f"""DROP TABLE IF EXISTS `{self.table_name}`"""
            connection.execute(sqlalchemy.text(stmt))

            connection.commit()

        self.close()

    def clear(self) -> None:
        self._initialize()

        with self._engine.connect() as connection:
            stmt = f"""DELETE FROM `{self.table_name}`"""
            connection.execute(sqlalchemy.text(stmt))

            connection.commit()


def _meets_min_server_version(version: str, min_version: str) -> bool:
    """Check if a MariaDB server version meets minimum required version.

    Args:
        version: Version string from MariaDB server (e.g. "11.7.1-MariaDB-ubu2404")
        min_version: Minimum required version string (e.g. "11.7.1")

    Returns:
        bool: True if version >= min_version, False otherwise
    """
    version = version.split("-")[0]
    version_parts = [int(x) for x in version.split(".")]
    min_version_parts = [int(x) for x in min_version.split(".")]
    return version_parts >= min_version_parts
