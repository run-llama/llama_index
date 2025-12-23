import json
import logging
import re
from typing import Any, Dict, List, NamedTuple, Optional, Literal

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
from mysql.connector.errors import Error as MySQLError
from sqlalchemy import (
    create_engine,
    Column,
    String,
    JSON,
    select,
    text as sql_text,
    and_,
    or_,
)
from sqlalchemy.orm import sessionmaker, declarative_base, registry


class DBEmbeddingRow(NamedTuple):
    node_id: str
    text: str
    metadata: dict
    similarity: float


_logger = logging.getLogger(__name__)


def get_data_model(base, table_name, embed_dim):
    """Create a dynamic SQLAlchemy model for the vector store table."""
    # Define the table structure
    mapper_registry = registry()

    # Create the table dynamically using Core Table
    vector_table = mapper_registry.metadata.tables.get(table_name)
    if vector_table is None:
        from sqlalchemy import Table
        from sqlalchemy.dialects.mysql import LONGTEXT

        vector_table = Table(
            table_name,
            mapper_registry.metadata,
            Column("id", String(36), primary_key=True),
            Column("node_id", String(255), nullable=False),
            Column("text", LONGTEXT),
            Column("metadata", JSON),
            Column(
                "embedding", String(2000), nullable=False
            ),  # Store as JSON string for now
            extend_existing=True,
        )

    return vector_table


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

    _engine: Any = PrivateAttr()
    _session: Any = PrivateAttr()
    _base: Any = PrivateAttr()
    _table: Any = PrivateAttr()
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

        # Initialize SQLAlchemy components
        self._base = declarative_base()
        self._table = get_data_model(self._base, self.table_name, self.embed_dim)
        connection_string = (
            f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        )
        self._engine = create_engine(connection_string, echo=False)
        self._session = sessionmaker(bind=self._engine)

        self._initialize()

    def close(self) -> None:
        if not self._is_initialized:
            return
        if hasattr(self._engine, "dispose"):
            self._engine.dispose()
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
        """Return the SQLAlchemy engine."""
        if not self._is_initialized:
            return None
        return self._engine

    def _check_vector_support(self) -> None:
        """Check if the MySQL server supports vector operations."""
        with self._session() as session:
            try:
                # Check MySQL version
                # Try to execute a simple vector function to verify support
                result = session.execute(
                    sql_text(
                        "SELECT VEC_FromText('[1,2,3]') IS NOT NULL as vector_support"
                    )
                )
                vector_result = result.fetchone()
                if not vector_result or not vector_result[0]:
                    raise ValueError(
                        "RDS MySQL Vector functions are not available."
                        " Please ensure you're using RDS MySQL 8.0.36+ with Vector support."
                    )

                # Check rds_release_date >= 20251031
                result = session.execute(
                    sql_text("SHOW VARIABLES LIKE 'rds_release_date'")
                )
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

            except MySQLError as e:
                if "FUNCTION" in str(e) and "VEC_FromText" in str(e):
                    raise ValueError(
                        "RDS MySQL Vector functions are not available."
                        " Please ensure you're using RDS MySQL 8.0.36+ with Vector support."
                    ) from e
                raise

    def _create_table_if_not_exists(self) -> None:
        with self._session() as session:
            # Create table with VECTOR data type for Alibaba Cloud MySQL
            stmt = sql_text(f"""
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
        with self._session() as session:
            # Build query using SQLAlchemy core
            query = select(self._table.c.text, self._table.c.metadata)

            conditions = []
            if node_ids:
                conditions.append(self._table.c.node_id.in_(node_ids))

            if filters:
                filter_conditions = self._filters_to_sqlalchemy_conditions(filters)
                if filter_conditions is not None:
                    conditions.append(filter_conditions)

            if conditions:
                query = query.where(and_(*conditions))

            results = session.execute(query).fetchall()

            for item in results:
                node = metadata_dict_to_node(json.loads(item.metadata))
                node.set_content(str(item.text))
                nodes.append(node)

        return nodes

    def _node_to_table_row(self, node: BaseNode) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "embedding": json.dumps(node.get_embedding()),
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
        with self._session() as session:
            for node in nodes:
                ids.append(node.node_id)
                item = self._node_to_table_row(node)

                # Insert using raw SQL to leverage MySQL vector functions
                stmt = sql_text(f"""
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
                        "embedding": item["embedding"],
                        "metadata": json.dumps(item["metadata"]),
                    },
                )
            session.commit()

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
        elif operator == FilterOperator.CONTAINS:
            return "JSON_CONTAINS"
        elif operator == FilterOperator.TEXT_MATCH:
            return "LIKE"
        elif operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
            return "ILIKE"  # Note: MySQL doesn't have ILIKE, using LIKE with LOWER
        elif operator == FilterOperator.IS_EMPTY:
            return "IS NULL or = ''"
        else:
            _logger.warning("Unsupported operator: %s, fallback to '='", operator)
            return "="

    def _filters_to_sqlalchemy_conditions(self, filters: MetadataFilters):
        """Convert filters to SQLAlchemy text conditions."""
        from sqlalchemy import text as sa_text

        if filters.condition == FilterCondition.OR:
            condition_func = or_
        elif filters.condition == FilterCondition.AND:
            condition_func = and_
        else:
            raise ValueError(
                f"Unsupported condition: {filters.condition}. "
                f"Must be one of {[FilterCondition.AND, FilterCondition.OR]}"
            )

        conditions = []
        for filter_ in filters.filters:
            if isinstance(filter_, MetadataFilter):
                # Build the condition based on the operator
                op = self._to_mysql_operator(filter_.operator)

                if op == "IN":
                    # Handle IN operator
                    if isinstance(filter_.value, list):
                        values_str = ",".join([f"'{v!s}'" for v in filter_.value])
                        condition = sa_text(
                            f"JSON_VALUE(metadata, '$.{filter_.key}') IN ({values_str})"
                        )
                    else:
                        condition = sa_text(
                            f"JSON_VALUE(metadata, '$.{filter_.key}') = '{filter_.value}'"
                        )
                elif op == "NOT IN":
                    # Handle NOT IN operator
                    if isinstance(filter_.value, list):
                        values_str = ",".join([f"'{v!s}'" for v in filter_.value])
                        condition = sa_text(
                            f"JSON_VALUE(metadata, '$.{filter_.key}') NOT IN ({values_str})"
                        )
                    else:
                        condition = sa_text(
                            f"JSON_VALUE(metadata, '$.{filter_.key}') != '{filter_.value}'"
                        )
                elif op == "JSON_CONTAINS":
                    # Handle contains operator for JSON arrays
                    condition = sa_text(
                        f"JSON_CONTAINS(metadata->'{filter_.key}', '\"{filter_.value}\"')"
                    )
                elif op in ["LIKE", "ILIKE"]:
                    # Handle text matching (ILIKE doesn't exist in MySQL, use LIKE)
                    if op == "ILIKE":
                        condition = sa_text(
                            f"LOWER(JSON_VALUE(metadata, '$.{filter_.key}')) LIKE LOWER('%{filter_.value}%')"
                        )
                    else:
                        condition = sa_text(
                            f"JSON_VALUE(metadata, '$.{filter_.key}') LIKE '%{filter_.value}%'"
                        )
                elif op == "IS NULL or = ''":
                    # Handle empty check
                    condition = sa_text(
                        f"(JSON_VALUE(metadata, '$.{filter_.key}') IS NULL OR JSON_VALUE(metadata, '$.{filter_.key}') = '')"
                    )
                else:
                    # Handle standard comparison operators
                    condition = sa_text(
                        f"JSON_VALUE(metadata, '$.{filter_.key}') {op} '{filter_.value}'"
                    )

                conditions.append(condition)
            elif isinstance(filter_, MetadataFilters):
                # Recursively process nested filters
                subcondition = self._filters_to_sqlalchemy_conditions(filter_)
                if subcondition is not None:
                    conditions.append(subcondition)
            else:
                raise ValueError(
                    f"Unsupported filter type: {type(filter_)}. Must be one of {MetadataFilter}, {MetadataFilters}"
                )

        if conditions:
            return condition_func(*conditions)
        return None

    def _filters_to_where_clause(self, filters: MetadataFilters) -> tuple[str, list]:
        """Convert filters to WHERE clause and parameters (for backward compatibility)."""
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
                clause = self._build_filter_clause(filter_)
                clauses.append(clause)
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

    def _build_filter_clause(self, filter_: MetadataFilter) -> str:
        """Build filter clause (for backward compatibility)."""
        values = []

        if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
            filter_value = f"({','.join(['%s'] * len(filter_.value))})"
            values.extend(filter_.value)
        elif isinstance(filter_.value, (list, tuple)):
            filter_value = f"({','.join(['%s'] * len(filter_.value))})"
            values.extend(filter_.value)
        else:
            filter_value = "%s"
            values.append(filter_.value)

        return f"JSON_VALUE(metadata, '$.{filter_.key}') {self._to_mysql_operator(filter_.operator)} {filter_value}"

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
            raise NotImplementedError(f"Query mode {query.mode!r} not available.")

        self._initialize()

        # Using specified distance function for vector similarity search
        distance_func = (
            "VEC_DISTANCE_COSINE"
            if self.distance_method == "COSINE"
            else "VEC_DISTANCE_EUCLIDEAN"
        )

        # Build the query using SQLAlchemy's select construct for proper parameter binding
        from sqlalchemy import select

        # Create the base select statement
        stmt = select(
            self._table.c.node_id,
            self._table.c.text,
            self._table.c.embedding,
            self._table.c.metadata,
        ).add_columns(
            sql_text(
                f"{distance_func}(embedding, VEC_FromText(:embedding)) AS distance"
            )
        )

        # Add parameters
        params = {
            "embedding": json.dumps(query.query_embedding),
        }

        # Add WHERE clause if filters exist using SQLAlchemy's text conditions
        if query.filters:
            # Use SQLAlchemy text conditions for proper parameter binding
            filter_conditions = self._filters_to_sqlalchemy_conditions(query.filters)
            if filter_conditions is not None:
                # For vector distance calculation, we need to use raw SQL with proper parameter binding
                # Build a complete SQL string with named parameters
                base_query = f"""
                SELECT
                    node_id,
                    text,
                    embedding,
                    metadata,
                    {distance_func}(embedding, VEC_FromText(:embedding)) AS distance
                FROM `{self.table_name}`
                """

                # Get filter conditions using SQLAlchemy expressions
                sa_filter_conditions = self._filters_to_sqlalchemy_conditions(
                    query.filters
                )
                if sa_filter_conditions is not None:
                    # Convert the SQLAlchemy condition to string and ensure proper parameter binding
                    # Build the filter part with named parameters
                    where_clause, filter_params = (
                        self._build_where_clause_with_named_params(
                            query.filters, params
                        )
                    )
                    if where_clause:
                        base_query += (
                            f"WHERE {where_clause} ORDER BY distance LIMIT :limit"
                        )
                        params.update(filter_params)
                        params["limit"] = query.similarity_top_k
                    else:
                        base_query += "ORDER BY distance LIMIT :limit"
                        params["limit"] = query.similarity_top_k
                else:
                    base_query += "ORDER BY distance LIMIT :limit"
                    params["limit"] = query.similarity_top_k
            else:
                base_query += "ORDER BY distance LIMIT :limit"
                params["limit"] = query.similarity_top_k
        else:
            base_query += "ORDER BY distance LIMIT :limit"
            params["limit"] = query.similarity_top_k

        stmt = sql_text(base_query)

        with self._session() as session:
            result = session.execute(stmt, params)
            results = result.fetchall()

        rows = []
        for item in results:
            rows.append(
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=json.loads(item.metadata),
                    similarity=(1 - item.distance) if item.distance is not None else 0,
                )
            )

        return self._db_rows_to_query_result(rows)

    def _build_where_clause_with_named_params(
        self, filters: MetadataFilters, existing_params: dict
    ) -> tuple[str, dict]:
        """Build WHERE clause with named parameters to avoid mixing %s and :param formats."""
        if not filters:
            return "", {}

        # Get the where clause and values using the existing method
        where_clause, values = self._filters_to_where_clause(filters)

        if not where_clause:
            return "", {}

        # Create new parameter names to avoid conflicts
        filter_params = {}
        param_index = len(existing_params)  # Start from current param count

        # Replace %s placeholders with named parameters
        for i, value in enumerate(values):
            param_name = f"filter_param_{param_index + i}"
            filter_params[param_name] = value
            # Replace each %s occurrence one by one
            where_clause = where_clause.replace("%s", f":{param_name}", 1)

        return where_clause, filter_params

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._initialize()

        with self._session() as session:
            # Delete based on ref_doc_id in metadata
            stmt = sql_text(
                f"""DELETE FROM `{self.table_name}` WHERE JSON_EXTRACT(metadata, '$.ref_doc_id') = :ref_doc_id"""
            )
            session.execute(stmt, {"ref_doc_id": ref_doc_id})
            session.commit()

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        self._initialize()

        with self._session() as session:
            base_query = f"DELETE FROM `{self.table_name}`"
            conditions = []
            params = {}

            if node_ids:
                # Using raw SQL for IN clause with named parameters
                placeholders = ",".join([f":node_id_{i}" for i in range(len(node_ids))])
                conditions.append(f"node_id IN ({placeholders})")
                for i, node_id in enumerate(node_ids):
                    params[f"node_id_{i}"] = node_id

            if filters:
                # Use the improved method for filter parameter binding
                where_clause, filter_params = (
                    self._build_where_clause_with_named_params(filters, params)
                )
                if where_clause:
                    conditions.append(f"({where_clause})")
                    params.update(filter_params)

            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)
                stmt = sql_text(base_query + where_clause)
                session.execute(stmt, params)
                session.commit()
            else:
                # If no conditions are provided, don't delete all records
                raise ValueError(
                    "Either node_ids or filters must be provided for delete_nodes"
                )

    def count(self) -> int:
        self._initialize()

        with self._session() as session:
            stmt = sql_text(f"""SELECT COUNT(*) as count FROM `{self.table_name}`""")
            result = session.execute(stmt).fetchone()

        return result[0] if result else 0

    def drop(self) -> None:
        self._initialize()

        with self._session() as session:
            stmt = sql_text(f"""DROP TABLE IF EXISTS `{self.table_name}`""")
            session.execute(stmt)
            session.commit()

        self.close()

    def clear(self) -> None:
        self._initialize()

        with self._session() as session:
            stmt = sql_text(f"""DELETE FROM `{self.table_name}`""")
            session.execute(stmt)
            session.commit()
