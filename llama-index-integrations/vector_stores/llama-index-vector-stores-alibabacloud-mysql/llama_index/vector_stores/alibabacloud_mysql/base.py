"""Alibaba Cloud MySQL Vector Store."""

import json
import logging
import re
from typing import Any, Dict, List, NamedTuple, Optional, Literal, Sequence
from urllib.parse import quote_plus

import sqlalchemy
import sqlalchemy.ext.asyncio
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


class AlibabaCloudMySQLVectorStore(BasePydanticVectorStore):
    """
    Alibaba Cloud MySQL Vector Store.

    Examples:
        ```python
        from llama_index.vector_stores.alibabacloud_mysql import AlibabaCloudMySQLVectorStore

        # Create AlibabaCloudMySQLVectorStore instance
        vector_store = AlibabaCloudMySQLVectorStore(
            table_name="llama_index_vectorstore",
            host="localhost",
            port=3306,
            user="llamaindex",
            password="password",
            database="vectordb",
            embed_dim=1536,  # OpenAI embedding dimension
            default_m=6,
            distance_method="COSINE"
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = False

    connection_string: str
    table_name: str = "llama_index_table"
    database: str
    embed_dim: int = 1536
    default_m: int = 6
    distance_method: Literal["EUCLIDEAN", "COSINE"] = "COSINE"
    perform_setup: bool = True
    debug: bool = False

    _engine: Any = PrivateAttr()
    _async_engine: Any = PrivateAttr()
    _session: Any = PrivateAttr()
    _async_session: Any = PrivateAttr()
    _table_class: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

    def _validate_identifier(self, name: str) -> str:
        # 只允许字母、数字、下划线（符合 SQL 标识符规范）
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid identifier: {name}")
        return name

    def _validate_positive_int(self, value: int, param_name: str) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Expected positive int for {param_name}, got {value}")
        return value

    def _validate_table_name(self, table_name: str) -> str:
        return self._validate_identifier(table_name)

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        table_name: str = "llama_index_table",
        embed_dim: int = 1536,
        default_m: int = 6,
        distance_method: Literal["EUCLIDEAN", "COSINE"] = "COSINE",
        perform_setup: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Constructor.

        Args:
            host (str): Host of Alibaba Cloud MySQL connection.
            port (int): Port of Alibaba Cloud MySQL connection.
            user (str): Alibaba Cloud MySQL username.
            password (str): Alibaba Cloud MySQL password.
            database (str): Alibaba Cloud MySQL DB name.
            table_name (str, optional): Table name for the vector store. Defaults to "llama_index_table".
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            default_m (int, optional): Default M value for the vector index. Defaults to 6.
            distance_method (Literal["EUCLIDEAN", "COSINE"], optional): Vector distance type. Defaults to COSINE.
            perform_setup (bool, optional): If DB should be set up. Defaults to True.
            debug (bool, optional): If debug logging should be enabled. Defaults to False.

        """
        # Validate table_name, embed_dim, and default_m
        self._validate_table_name(table_name)
        self._validate_positive_int(embed_dim, "embed_dim")
        self._validate_positive_int(default_m, "default_m")

        # Create connection string
        password_safe = quote_plus(password)
        connection_string = (
            f"mysql+pymysql://{user}:{password_safe}@{host}:{port}/{database}"
        )

        super().__init__(
            connection_string=connection_string,
            table_name=table_name,
            database=database,
            embed_dim=embed_dim,
            default_m=default_m,
            distance_method=distance_method,
            perform_setup=perform_setup,
            debug=debug,
        )

        # Private attrs
        self._engine = None
        self._async_engine = None
        self._session = None
        self._async_session = None
        self._table_class = None
        self._is_initialized = False

        self._initialize()

    @classmethod
    def class_name(cls) -> str:
        return "AlibabaCloudMySQLVectorStore"

    @classmethod
    def from_params(
        cls,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        table_name: str = "llama_index_table",
        embed_dim: int = 1536,
        default_m: int = 6,
        distance_method: Literal["EUCLIDEAN", "COSINE"] = "COSINE",
        perform_setup: bool = True,
        debug: bool = False,
    ) -> "AlibabaCloudMySQLVectorStore":
        """
        Construct from params.

        Args:
            host (str): Host of Alibaba Cloud MySQL connection.
            port (int): Port of Alibaba Cloud MySQL connection.
            user (str): Alibaba Cloud MySQL username.
            password (str): Alibaba Cloud MySQL password.
            database (str): Alibaba Cloud MySQL DB name.
            table_name (str, optional): Table name for the vector store. Defaults to "llama_index_table".
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            default_m (int, optional): Default M value for the vector index. Defaults to 6.
            distance_method (Literal["EUCLIDEAN", "COSINE"], optional): Vector distance type. Defaults to COSINE.
            perform_setup (bool, optional): If DB should be set up. Defaults to True.
            debug (bool, optional): If debug logging should be enabled. Defaults to False.

        Returns:
            AlibabaCloudMySQLVectorStore: Instance of AlibabaCloudMySQLVectorStore constructed from params.

        """
        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            table_name=table_name,
            embed_dim=embed_dim,
            default_m=default_m,
            distance_method=distance_method,
            perform_setup=perform_setup,
            debug=debug,
        )

    @property
    def client(self) -> Any:
        """Return the SQLAlchemy engine."""
        if not self._is_initialized:
            return None
        return self._engine

    def _connect(self) -> None:
        """Create SQLAlchemy engines and sessions."""
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        # Create sync engine
        self._engine = create_engine(
            self.connection_string,
            echo=self.debug,
        )

        # Create async engine
        async_connection_string = self.connection_string.replace(
            "mysql+pymysql://", "mysql+aiomysql://"
        )
        self._async_engine = create_async_engine(
            async_connection_string,
            echo=self.debug,
        )

        # Create session makers
        self._session = sessionmaker(self._engine)
        self._async_session = sessionmaker(self._async_engine, class_=AsyncSession)

    def _check_vector_support(self) -> None:
        """Check if the MySQL server supports vector operations."""
        from sqlalchemy import text

        with self._session() as session:
            try:
                # Check MySQL version
                # Try to execute a simple vector function to verify support
                result = session.execute(
                    text("SELECT VEC_FromText('[1,2,3]') IS NOT NULL as vector_support")
                )
                vector_result = result.fetchone()
                if not vector_result or not vector_result[0]:
                    raise ValueError(
                        "RDS MySQL Vector functions are not available."
                        " Please ensure you're using RDS MySQL 8.0.36+ with Vector support."
                    )

                # Check rds_release_date >= 20251031
                result = session.execute(text("SHOW VARIABLES LIKE 'rds_release_date'"))
                rds_release_result = result.fetchone()

                if not rds_release_result:
                    raise ValueError(
                        "Unable to retrieve rds_release_date variable. "
                        "Your MySQL instance may not Alibaba Cloud RDS MySQL instance."
                    )

                rds_release_date = rds_release_result[1]
                if int(rds_release_date) < 20251031:
                    raise ValueError(
                        f"Alibaba Cloud MySQL rds_release_date must be 20251031 or later, found: {rds_release_date}."
                    )

            except Exception as e:
                if "FUNCTION" in str(e) and "VEC_FromText" in str(e):
                    raise ValueError(
                        "RDS MySQL Vector functions are not available."
                        " Please ensure you're using RDS MySQL 8.0.36+ with Vector support."
                    ) from e
                raise

    def _create_table_if_not_exists(self) -> None:
        from sqlalchemy import text

        with self._session() as session:
            # Create table with VECTOR data type for Alibaba Cloud MySQL
            stmt = text(f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                id VARCHAR(36) PRIMARY KEY,
                node_id VARCHAR(255) NOT NULL,
                text LONGTEXT,
                metadata JSON,
                embedding VECTOR({self.embed_dim}) NOT NULL,
                INDEX `node_id_index` (node_id),
                VECTOR INDEX (embedding) M={self.default_m} DISTANCE={self.distance_method}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            session.execute(stmt)
            session.commit()

    def _initialize(self) -> None:
        if not self._is_initialized:
            self._connect()
            if self.perform_setup:
                self._check_vector_support()
                self._create_table_if_not_exists()
            self._is_initialized = True

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

    def _to_mysql_operator(self, operator: FilterOperator) -> str:
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

    def _build_filter_clause(
        self, filter_: MetadataFilter, global_param_counter: List[int]
    ) -> tuple[str, dict]:
        params = {}

        if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
            # For IN/NIN operators, we need multiple placeholders
            placeholders = []
            for i in range(len(filter_.value)):
                param_name = f"param_{global_param_counter[0]}"
                global_param_counter[0] += 1
                placeholders.append(f":{param_name}")
                params[param_name] = filter_.value[i]
            filter_value = f"({','.join(placeholders)})"
        elif isinstance(filter_.value, (list, tuple)):
            # For list/tuple values, we also need multiple placeholders
            placeholders = []
            for i in range(len(filter_.value)):
                param_name = f"param_{global_param_counter[0]}"
                global_param_counter[0] += 1
                placeholders.append(f":{param_name}")
                params[param_name] = filter_.value[i]
            filter_value = f"({','.join(placeholders)})"
        else:
            # For single value, create a single parameter
            param_name = f"param_{global_param_counter[0]}"
            global_param_counter[0] += 1
            filter_value = f":{param_name}"
            params[param_name] = filter_.value

        clause = f"JSON_VALUE(metadata, '$.{filter_.key}') {self._to_mysql_operator(filter_.operator)} {filter_value}"
        return clause, params

    def _filters_to_where_clause(
        self, filters: MetadataFilters, global_param_counter: List[int]
    ) -> tuple[str, dict]:
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
        all_params = {}

        for filter_ in filters.filters:
            if isinstance(filter_, MetadataFilter):
                clause, filter_params = self._build_filter_clause(
                    filter_, global_param_counter
                )
                clauses.append(clause)
                all_params.update(filter_params)
                continue

            if isinstance(filter_, MetadataFilters):
                subclause, subparams = self._filters_to_where_clause(
                    filter_, global_param_counter
                )
                if subclause:
                    clauses.append(f"({subclause})")
                    all_params.update(subparams)
                continue

            raise ValueError(
                f"Unsupported filter type: {type(filter_)}. Must be one of {MetadataFilter}, {MetadataFilters}"
            )

        return f" {conditions[filters.condition]} ".join(clauses), all_params

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

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes from vector store."""
        self._initialize()

        nodes: List[BaseNode] = []
        with self._session() as session:
            if node_ids:
                # Using parameterized query to prevent SQL injection
                placeholders = ",".join([f":node_id_{i}" for i in range(len(node_ids))])
                params = {f"node_id_{i}": node_id for i, node_id in enumerate(node_ids)}
                stmt = sqlalchemy.text(
                    f"""SELECT text, metadata FROM `{self.table_name}` WHERE node_id IN ({placeholders})"""
                )
                result = session.execute(stmt, params)
            else:
                stmt = sqlalchemy.text(
                    f"""SELECT text, metadata FROM `{self.table_name}`"""
                )
                result = session.execute(stmt)

            for item in result:
                node = metadata_dict_to_node(
                    json.loads(item[1]) if isinstance(item[1], str) else item[1]
                )
                node.set_content(str(item[0]))
                nodes.append(node)

        return nodes

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        self._initialize()

        ids = []
        with self._session() as session:
            for node in nodes:
                ids.append(node.node_id)
                item = self._node_to_table_row(node)

                stmt = sqlalchemy.text(f"""
                INSERT INTO `{self.table_name}` (id, node_id, text, embedding, metadata)
                VALUES (
                    UUID(),
                    :node_id,
                    :text,
                    VEC_FromText(:embedding),
                    :metadata
                )
                ON DUPLICATE KEY UPDATE
                    text = VALUES(text),
                    embedding = VALUES(embedding),
                    metadata = VALUES(metadata)
                """)

                session.execute(
                    stmt,
                    {
                        "node_id": item["node_id"],
                        "text": item["text"],
                        "embedding": json.dumps(item["embedding"]),
                        "metadata": json.dumps(item["metadata"]),
                    },
                )
            session.commit()

        return ids

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Async wrapper around :meth:`add`.
        """
        self._initialize()

        if not nodes:
            return []

        ids: List[str] = []
        async with self._async_session() as session:
            for node in nodes:
                ids.append(node.node_id)
                item = self._node_to_table_row(node)

                stmt = sqlalchemy.text(f"""
                INSERT INTO `{self.table_name}` (id, node_id, text, embedding, metadata)
                VALUES (
                    UUID(),
                    :node_id,
                    :text,
                    VEC_FromText(:embedding),
                    :metadata
                )
                ON DUPLICATE KEY UPDATE
                    text = VALUES(text),
                    embedding = VALUES(embedding),
                    metadata = VALUES(metadata)
                """)

                await session.execute(
                    stmt,
                    {
                        "node_id": item["node_id"],
                        "text": item["text"],
                        "embedding": json.dumps(item["embedding"]),
                        "metadata": json.dumps(item["metadata"]),
                    },
                )
            await session.commit()

        return ids

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"Query mode {query.mode} not available.")

        self._initialize()

        # Using specified distance function for vector similarity search
        distance_func = (
            "VEC_DISTANCE_COSINE"
            if self.distance_method == "COSINE"
            else "VEC_DISTANCE_EUCLIDEAN"
        )

        where_clause = ""
        params = {
            "query_embedding": json.dumps(query.query_embedding),
            "limit": query.similarity_top_k,
        }

        if query.filters:
            # Use a global counter to ensure unique parameter names
            global_param_counter = [0]  # Use a list to make it mutable
            where_clause, filter_params = self._filters_to_where_clause(
                query.filters, global_param_counter
            )
            where_clause = f"WHERE {where_clause}"
            params.update(filter_params)

        stmt = sqlalchemy.text(f"""
        SELECT
            node_id,
            text,
            embedding,
            metadata,
            {distance_func}(embedding, VEC_FromText(:query_embedding)) AS distance
        FROM `{self.table_name}`
        {where_clause}
        ORDER BY distance
        LIMIT :limit
        """)

        with self._session() as session:
            result = session.execute(stmt, params)
            results = result.fetchall()

        rows = []
        for item in results:
            rows.append(
                DBEmbeddingRow(
                    node_id=item[0],
                    text=item[1],
                    metadata=json.loads(item[3])
                    if isinstance(item[3], str)
                    else item[3],
                    similarity=(1 - item[4]) if item[4] is not None else 0,
                )
            )

        return self._db_rows_to_query_result(rows)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async wrapper around :meth:`query`."""
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"Query mode {query.mode} not available.")

        self._initialize()

        # Using specified distance function for vector similarity search
        distance_func = (
            "VEC_DISTANCE_COSINE"
            if self.distance_method == "COSINE"
            else "VEC_DISTANCE_EUCLIDEAN"
        )

        where_clause = ""
        params = {
            "query_embedding": json.dumps(query.query_embedding),
            "limit": query.similarity_top_k,
        }

        if query.filters:
            # Use a global counter to ensure unique parameter names
            global_param_counter = [0]  # Use a list to make it mutable
            where_clause, filter_params = self._filters_to_where_clause(
                query.filters, global_param_counter
            )
            where_clause = f"WHERE {where_clause}"
            params.update(filter_params)

        stmt = sqlalchemy.text(f"""
        SELECT
            node_id,
            text,
            embedding,
            metadata,
            {distance_func}(embedding, VEC_FromText(:query_embedding)) AS distance
        FROM `{self.table_name}`
        {where_clause}
        ORDER BY distance
        LIMIT :limit
        """)

        async with self._async_session() as session:
            result = await session.execute(stmt, params)
            results = result.fetchall()

        rows = []
        for item in results:
            rows.append(
                DBEmbeddingRow(
                    node_id=item[0],
                    text=item[1],
                    metadata=json.loads(item[3])
                    if isinstance(item[3], str)
                    else item[3],
                    similarity=(1 - item[4]) if item[4] is not None else 0,
                )
            )

        return self._db_rows_to_query_result(rows)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._initialize()

        with self._session() as session:
            # Delete based on ref_doc_id in metadata
            stmt = sqlalchemy.text(
                f"""DELETE FROM `{self.table_name}` WHERE JSON_EXTRACT(metadata, '$.ref_doc_id') = :doc_id"""
            )
            session.execute(stmt, {"doc_id": ref_doc_id})
            session.commit()

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async wrapper around :meth:`delete`."""
        self._initialize()

        async with self._async_session() as session:
            # Delete based on ref_doc_id in metadata
            stmt = sqlalchemy.text(
                f"""DELETE FROM `{self.table_name}` WHERE JSON_EXTRACT(metadata, '$.ref_doc_id') = :doc_id"""
            )
            await session.execute(stmt, {"doc_id": ref_doc_id})
            await session.commit()

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        self._initialize()

        with self._session() as session:
            if node_ids:
                # Using parameterized query to prevent SQL injection
                placeholders = ",".join([f":node_id_{i}" for i in range(len(node_ids))])
                params = {f"node_id_{i}": node_id for i, node_id in enumerate(node_ids)}
                stmt = sqlalchemy.text(
                    f"""DELETE FROM `{self.table_name}` WHERE node_id IN ({placeholders})"""
                )
                session.execute(stmt, params)
                session.commit()

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Async wrapper around :meth:`delete_nodes`."""
        self._initialize()

        async with self._async_session() as session:
            if node_ids:
                # Using parameterized query to prevent SQL injection
                placeholders = ",".join([f":node_id_{i}" for i in range(len(node_ids))])
                params = {f"node_id_{i}": node_id for i, node_id in enumerate(node_ids)}
                stmt = sqlalchemy.text(
                    f"""DELETE FROM `{self.table_name}` WHERE node_id IN ({placeholders})"""
                )
                await session.execute(stmt, params)
                await session.commit()

    def count(self) -> int:
        self._initialize()

        with self._session() as session:
            stmt = sqlalchemy.text(
                f"""SELECT COUNT(*) as count FROM `{self.table_name}`"""
            )
            result = session.execute(stmt)
            row = result.fetchone()

        return row[0] if row else 0

    def drop(self) -> None:
        self._initialize()

        with self._session() as session:
            stmt = sqlalchemy.text(f"""DROP TABLE IF EXISTS `{self.table_name}`""")
            session.execute(stmt)
            session.commit()

        self.close()

    def clear(self) -> None:
        self._initialize()

        with self._session() as session:
            stmt = sqlalchemy.text(f"""DELETE FROM `{self.table_name}`""")
            session.execute(stmt)
            session.commit()

    async def aclear(self) -> None:
        """Async wrapper around :meth:`clear`."""
        self._initialize()

        async with self._async_session() as session:
            stmt = sqlalchemy.text(f"""DELETE FROM `{self.table_name}`""")
            await session.execute(stmt)
            await session.commit()

    def close(self) -> None:
        if not self._is_initialized:
            return
        if self._engine:
            self._engine.dispose()
        if self._async_engine:
            import asyncio

            try:
                # Try to run the async disposal
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    asyncio.run(self._async_engine.dispose())
                else:
                    # If already in a running loop, create a new thread to run the disposal
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, self._async_engine.dispose()
                        )
                        future.result()
            except RuntimeError:
                # If no event loop exists, create one
                asyncio.run(self._async_engine.dispose())
        self._is_initialized = False

    async def aclose(self) -> None:
        if not self._is_initialized:
            return
        if self._engine:
            self._engine.dispose()
        if self._async_engine:
            await self._async_engine.dispose()
        self._is_initialized = False
