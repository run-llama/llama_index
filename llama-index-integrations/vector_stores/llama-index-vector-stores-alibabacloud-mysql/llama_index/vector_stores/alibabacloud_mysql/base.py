"""Alibaba Cloud MySQL Vector Store."""

import re
import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Literal
from contextlib import contextmanager

import mysql.connector
import mysql.connector.pooling
from mysql.connector.connection import MySQLConnection
from mysql.connector.cursor import MySQLCursor
from mysql.connector.errors import Error as MySQLError

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

    table_name: str = "llama_index_table"
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"
    max_connection: int = 10
    embed_dim: int = 1536
    default_m: int = 6
    distance_method: Literal["EUCLIDEAN", "COSINE"]
    perform_setup: bool = True

    _pool: mysql.connector.pooling.MySQLConnectionPool = PrivateAttr()
    _table_name: str = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)

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
        distance_method: str = "COSINE",
        perform_setup: bool = True,
        charset: str = "utf8mb4",
        max_connection: int = 10,
    ) -> None:
        """
        Constructor.

        Args:
            host (str): Host of Alibaba Cloud MySQL connection.
            port (int): Port of Alibaba Cloud MySQL connection.
            user (str): Alibaba Cloud MySQL username.
            password (str): Alibaba Cloud MySQL password.
            database (str): Alibaba Cloud MySQL DB name.
            table_name (str, optional): Table name for the vector store. Must contain only letters, numbers, underscores, and hyphens. Defaults to "llama_index_table".
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            default_m (int, optional): Default M value for the vector index. Defaults to 6.
            distance_method (str, optional): Vector distance type. Defaults to COSINE.
            perform_setup (bool, optional): If DB should be set up. Defaults to True.
            charset (str, optional): Character set for the connection. Defaults to utf8mb4.
            max_connection (int, optional): Maximum number of connections in the pool. Defaults to 10.

        """
        # Validate table_name to prevent SQL injection
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*$", table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        super().__init__(
            table_name=table_name,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset=charset,
            max_connection=max_connection,
            embed_dim=embed_dim,
            default_m=default_m,
            distance_method=distance_method,
            perform_setup=perform_setup,
        )

        self._pool = self._create_connection_pool()
        self._initialize()

    def close(self) -> None:
        if not self._is_initialized:
            return
        # Note: MySQLConnectionPool doesn't have explicit dispose/close method
        self._is_initialized = False

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
        distance_method: str = "COSINE",
        perform_setup: bool = True,
        charset: str = "utf8mb4",
        max_connection: int = 10,
    ) -> "AlibabaCloudMySQLVectorStore":
        """
        Construct from params.

        Args:
            host (str): Host of Alibaba Cloud MySQL connection.
            port (int): Port of Alibaba Cloud MySQL connection.
            user (str): Alibaba Cloud MySQL username.
            password (str): Alibaba Cloud MySQL password.
            database (str): Alibaba Cloud MySQL DB name.
            table_name (str, optional): Table name for the vector store. Must contain only letters, numbers, underscores, and hyphens. Defaults to "llama_index_table".
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            default_m (int, optional): Default M value for the vector index. Defaults to 6.
            distance_method (str, optional): Vector distance type. Defaults to COSINE.
            perform_setup (bool, optional): If DB should be set up. Defaults to True.
            charset (str, optional): Character set for the connection. Defaults to utf8mb4.
            max_connection (int, optional): Maximum number of connections in the pool. Defaults to 10.

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
            charset=charset,
            max_connection=max_connection,
        )

    @property
    def client(self) -> Any:
        """Return the MySQL connection pool."""
        if not self._is_initialized:
            return None
        return self._pool

    def _create_connection_pool(self) -> mysql.connector.pooling.MySQLConnectionPool:
        """Create connection pool using mysql-connector-python pooling."""
        pool_config: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
            "autocommit": True,
            "pool_name": f"pool_{self.table_name}",
            "pool_size": self.max_connection,
            "pool_reset_session": True,
        }
        return mysql.connector.pooling.MySQLConnectionPool(**pool_config)

    @contextmanager
    def _get_cursor(self):
        """Context manager to get a cursor from the connection pool."""
        conn: Optional[MySQLConnection] = None
        cursor: Optional[MySQLCursor] = None
        try:
            conn = self._pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            yield cursor
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _check_vector_support(self) -> None:
        """Check if the MySQL server supports vector operations."""
        try:
            with self._get_cursor() as cur:
                # Check MySQL version
                # Try to execute a simple vector function to verify support
                cur.execute(
                    "SELECT VEC_FromText('[1,2,3]') IS NOT NULL as vector_support"
                )
                vector_result = cur.fetchone()
                if not vector_result or not vector_result.get("vector_support"):
                    raise ValueError(
                        "RDS MySQL Vector functions are not available."
                        " Please ensure you're using RDS MySQL 8.0.36+ with Vector support."
                    )

                # Check rds_release_date >= 20251031
                cur.execute("SHOW VARIABLES LIKE 'rds_release_date'")
                rds_release_result = cur.fetchone()

                if not rds_release_result:
                    raise ValueError(
                        "Unable to retrieve rds_release_date variable. "
                        "Your MySQL instance may not Alibaba Cloud RDS MySQL instance."
                    )

                rds_release_date = rds_release_result["Value"]
                if int(rds_release_date) < 20251031:
                    raise ValueError(
                        f"Alibaba Cloud MySQL rds_release_date must be 20251031 or later, found: {rds_release_date}."
                    )

        except MySQLError as e:
            if "FUNCTION" in str(e) and "VEC_FromText" in str(e):
                raise ValueError(
                    "RDS MySQL Vector functions are not available."
                    " Please ensure you're using RDS MySQL 8.0.36+ with Vector support."
                ) from e
            raise

    def _create_table_if_not_exists(self) -> None:
        with self._get_cursor() as cur:
            # Create table with VECTOR data type for Alibaba Cloud MySQL
            stmt = f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                id VARCHAR(36) PRIMARY KEY,
                node_id VARCHAR(255) NOT NULL,
                text LONGTEXT,
                metadata JSON,
                embedding VECTOR({self.embed_dim}) NOT NULL,
                INDEX `node_id_index` (node_id),
                VECTOR INDEX (embedding) M={self.default_m} DISTANCE={self.distance_method}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cur.execute(stmt)

    def _initialize(self) -> None:
        if not self._is_initialized:
            self._check_vector_support()
            if self.perform_setup:
                self._create_table_if_not_exists()
            self._is_initialized = True

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes from vector store."""
        self._initialize()

        nodes: List[BaseNode] = []
        with self._get_cursor() as cur:
            where_conditions = []
            params = []

            if node_ids:
                placeholders = ",".join(["%s"] * len(node_ids)) if node_ids else ""
                where_conditions.append(f"node_id IN ({placeholders})")
                params.extend(node_ids)

            if filters:
                filter_clause, filter_params = self._filters_to_where_clause(filters)
                where_conditions.append(f"({filter_clause})")
                params.extend(filter_params)

            where_clause = (
                " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            )
            stmt = f"""SELECT text, metadata FROM `{self.table_name}`{where_clause}"""

            if params:
                cur.execute(stmt, params)
            else:
                cur.execute(stmt)

            results = cur.fetchall()

            for item in results:
                node = metadata_dict_to_node(json.loads(item["metadata"]))
                node.set_content(str(item["text"]))
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
        with self._get_cursor() as cur:
            for node in nodes:
                ids.append(node.node_id)
                item = self._node_to_table_row(node)

                stmt = f"""
                INSERT INTO `{self.table_name}` (id, node_id, text, embedding, metadata)
                VALUES (
                    UUID(),
                    %(node_id)s,
                    %(text)s,
                    VEC_FromText(%(embedding)s),
                    %(metadata)s
                )
                ON DUPLICATE KEY UPDATE
                    text = VALUES(text),
                    embedding = VALUES(embedding),
                    metadata = VALUES(metadata)
                """

                cur.execute(
                    stmt,
                    {
                        "node_id": item["node_id"],
                        "text": item["text"],
                        "embedding": json.dumps(item["embedding"]),
                        "metadata": json.dumps(item["metadata"]),
                    },
                )

        return ids

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

    def _build_filter_clause(self, filter_: MetadataFilter) -> tuple[str, list]:
        values = []

        if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
            placeholders = ",".join(["%s"] * len(filter_.value))
            filter_value = f"({placeholders})"
            values.extend(filter_.value)
        elif isinstance(filter_.value, (list, tuple)):
            placeholders = ",".join(["%s"] * len(filter_.value))
            filter_value = f"({placeholders})"
            values.extend(filter_.value)
        else:
            filter_value = "%s"
            values.append(filter_.value)

        clause = f"JSON_VALUE(metadata, '$.{filter_.key}') {self._to_mysql_operator(filter_.operator)} {filter_value}"
        return clause, values

    def _filters_to_where_clause(self, filters: MetadataFilters) -> tuple[str, list]:
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
        values: List[Any] = []

        for filter_ in filters.filters:
            if isinstance(filter_, MetadataFilter):
                clause, filter_values = self._build_filter_clause(filter_)
                clauses.append(clause)
                values.extend(filter_values)
                continue

            if isinstance(filter_, MetadataFilters):
                subclause, subvalues = self._filters_to_where_clause(filter_)
                if subclause:
                    clauses.append(f"({subclause})")
                    values.extend(subvalues)
                continue

            raise ValueError(
                f"Unsupported filter type: {type(filter_)}. Must be one of {MetadataFilter}, {MetadataFilters}"
            )

        return f" {conditions[filters.condition]} ".join(clauses), values

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

        # Using specified distance function for vector similarity search
        distance_func = (
            "VEC_DISTANCE_COSINE"
            if self.distance_method == "COSINE"
            else "VEC_DISTANCE_EUCLIDEAN"
        )

        where_clause = ""
        values = []
        if query.filters:
            where_clause, values = self._filters_to_where_clause(query.filters)
            where_clause = f"WHERE {where_clause}"

        stmt = f"""
        SELECT
            node_id,
            text,
            embedding,
            metadata,
            {distance_func}(embedding, VEC_FromText(%s)) AS distance
        FROM `{self.table_name}`
        {where_clause}
        ORDER BY distance
        LIMIT %s
        """

        # Add query embedding and limit to values
        values = [json.dumps(query.query_embedding), *values, query.similarity_top_k]

        with self._get_cursor() as cur:
            cur.execute(stmt, values)
            results = cur.fetchall()

        rows = []
        for item in results:
            rows.append(
                DBEmbeddingRow(
                    node_id=item["node_id"],
                    text=item["text"],
                    metadata=json.loads(item["metadata"]),
                    similarity=(1 - item["distance"])
                    if item["distance"] is not None
                    else 0,
                )
            )

        return self._db_rows_to_query_result(rows)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._initialize()

        with self._get_cursor() as cur:
            # Delete based on ref_doc_id in metadata
            stmt = f"""DELETE FROM `{self.table_name}` WHERE JSON_EXTRACT(metadata, '$.ref_doc_id') = %s"""
            cur.execute(stmt, (ref_doc_id,))

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        self._initialize()

        with self._get_cursor() as cur:
            where_conditions = []
            params = []

            if node_ids:
                placeholders = ",".join(["%s"] * len(node_ids))
                where_conditions.append(f"node_id IN ({placeholders})")
                params.extend(node_ids)

            if filters:
                filter_clause, filter_params = self._filters_to_where_clause(filters)
                where_conditions.append(f"({filter_clause})")
                params.extend(filter_params)

            if where_conditions:
                where_clause = " WHERE " + " AND ".join(where_conditions)
                stmt = f"""DELETE FROM `{self.table_name}`{where_clause}"""
                cur.execute(stmt, params)
            else:
                # If no conditions are provided, don't delete all records
                raise ValueError(
                    "Either node_ids or filters must be provided for delete_nodes"
                )

    def count(self) -> int:
        self._initialize()

        with self._get_cursor() as cur:
            stmt = f"""SELECT COUNT(*) as count FROM `{self.table_name}`"""
            cur.execute(stmt)
            result = cur.fetchone()

        return result["count"] if result else 0

    def drop(self) -> None:
        self._initialize()

        with self._get_cursor() as cur:
            stmt = f"""DROP TABLE IF EXISTS `{self.table_name}`"""
            cur.execute(stmt)

        self.close()

    def clear(self) -> None:
        self._initialize()

        with self._get_cursor() as cur:
            stmt = f"""DELETE FROM `{self.table_name}`"""
            cur.execute(stmt)
