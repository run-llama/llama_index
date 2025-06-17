"""DuckDB vector store."""

from ast import Expression
import json
import logging
import operator as py_operator
from pathlib import Path
from typing import Any

import duckdb
from duckdb import (
    ColumnExpression,
    ConstantExpression,
    FunctionExpression,
    StarExpression,
    Expression,
    CaseExpression,
)
from duckdb.typing import FLOAT, INTEGER, VARCHAR, SQLNULL
from fsspec.utils import Sequence
from llama_index.core.bridge.pydantic import (
    PrivateAttr,
    StrictInt,
    StrictFloat,
    StrictStr,
)
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryResult,
    FilterCondition,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)

DEFAULT_TEXT_SEARCH_CONFIG = {
    "stemmer": "english",
    "stopwords": "english",
    "ignore": "(\\.|[^a-z])+",
    "strip_accents": True,
    "lower": True,
    "overwrite": False,
}

li_filter_to_py_operator = {
    FilterOperator.GT: py_operator.gt,
    FilterOperator.GTE: py_operator.ge,
    FilterOperator.LT: py_operator.lt,
    FilterOperator.LTE: py_operator.le,
    FilterOperator.EQ: py_operator.eq,
    FilterOperator.NE: py_operator.ne,
}

filter_value_type_to_duckdb_type = {
    StrictInt: INTEGER,
    StrictFloat: FLOAT,
    StrictStr: VARCHAR,
    int: INTEGER,
    float: FLOAT,
    str: VARCHAR,
    None: SQLNULL,
}


class DuckDBInvalidConfigurationError(Exception):
    def __init__(self, database_name: str, persist_dir: str):
        self.database_name = database_name
        self.persist_dir = persist_dir
        super().__init__(
            f"Invalid configuration for DuckDBVectorStore. database_name: {database_name}, persist_dir: {persist_dir}"
        )


class DuckDBTableNotInitializedError(Exception):
    def __init__(self, table_name: str):
        self.table_name = table_name
        super().__init__(f"Table {table_name} is not initialized.")


class DuckDBTableIncorrectColumnsError(Exception):
    def __init__(
        self, table_name: str, expected_columns: list[str], actual_columns: list[str]
    ):
        self.table_name = table_name
        self.expected_columns = expected_columns
        self.actual_columns = actual_columns
        super().__init__(
            f"Table {table_name} has incorrect columns. Expected {expected_columns}, got {actual_columns}."
        )


class DuckDBVectorStore(BasePydanticVectorStore):
    """
    DuckDB vector store.

    In this vector store, embeddings are stored within a DuckDB database.

    During query time, the index uses DuckDB to query for the top
    k most similar nodes.

    Examples:
        `pip install llama-index-vector-stores-duckdb`

        ```python
        from llama_index.vector_stores.duckdb import DuckDBVectorStore

        # in-memory
        vector_store = DuckDBVectorStore()

        # persist to disk
        vector_store = DuckDBVectorStore("pg.duckdb", persist_dir="./persist/")
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    database_name: str
    table_name: str
    # schema_name: Optional[str] # TODO: support schema name
    embed_dim: int | None
    # hybrid_search: Optional[bool] # TODO: support hybrid search
    text_search_config: dict | None
    persist_dir: str

    _conn: duckdb.DuckDBPyConnection = PrivateAttr()
    _table: duckdb.DuckDBPyRelation | None = PrivateAttr(default=None)

    _is_initialized: bool = PrivateAttr(default=False)
    _database_path: str | None = PrivateAttr()

    def __init__(
        self,
        database_name: str = ":memory:",
        table_name: str = "documents",
        embed_dim: int | None = None,
        # https://duckdb.org/docs/extensions/full_text_search
        text_search_config: dict | None = None,
        persist_dir: str = "./storage",
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Init params."""
        fields = {
            "database_name": database_name,
            "table_name": table_name,
            "embed_dim": embed_dim,
            "text_search_config": text_search_config,
            "persist_dir": persist_dir,
        }

        super().__init__(stores_text=True, **fields)

        self._connect()
        self._initialize()

    @classmethod
    def from_local(
        cls,
        database_path: str,
        table_name: str = "documents",
        # schema_name: Optional[str] = "main",
        embed_dim: int | None = None,
        # hybrid_search: Optional[bool] = False,
        text_search_config: dict | None = DEFAULT_TEXT_SEARCH_CONFIG,
        **kwargs: Any,
    ) -> "DuckDBVectorStore":
        """Load a DuckDB vector store from a local file."""
        db_path = Path(database_path)

        return cls(
            database_name=db_path.name,
            table_name=table_name,
            embed_dim=embed_dim,
            text_search_config=text_search_config,
            persist_dir=str(db_path.parent),
            **kwargs,
        )

    @classmethod
    def from_params(
        cls,
        database_name: str = ":memory:",
        table_name: str = "documents",
        # schema_name: Optional[str] = "main",
        embed_dim: int | None = None,
        # hybrid_search: Optional[bool] = False,
        text_search_config: dict | None = DEFAULT_TEXT_SEARCH_CONFIG,
        persist_dir: str = "./storage",
        **kwargs: Any,
    ) -> "DuckDBVectorStore":
        return cls(
            database_name=database_name,
            table_name=table_name,
            # schema_name=schema_name,
            embed_dim=embed_dim,
            # hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            persist_dir=persist_dir,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "DuckDBVectorStore"

    @property
    def client(self) -> Any:
        """Return client."""
        return self._conn

    def _connect(self) -> None:
        """Connect to the DuckDB database -- create the data persistence directory if it doesn't exist."""
        database_connection = self.database_name

        if self.database_name != ":memory:":
            persist_path = Path(self.persist_dir)

            if not persist_path.exists():
                persist_path.mkdir(parents=True, exist_ok=True)

            database_connection = str(persist_path / self.database_name)

        self._conn = duckdb.connect(database_connection)

    def _initialize(self) -> None:
        """Initialize the DuckDB Database, extensions, and documents table."""
        home_dir = Path.home()
        self._conn.execute(f"SET home_directory='{home_dir}';")
        self._conn.install_extension("json")
        self._conn.load_extension("json")
        self._conn.install_extension("fts")
        self._conn.load_extension("fts")

        embedding_type = (
            f"FLOAT[{self.embed_dim}]" if self.embed_dim is not None else "FLOAT[]"
        )

        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name}  (
                node_id VARCHAR,
                text TEXT,
                embedding {embedding_type},
                metadata_ JSON
            );
        """)

        self._table = self._conn.table(self.table_name)

        required_columns = ["node_id", "text", "embedding", "metadata_"]
        table_columns = self._table.describe().columns

        for column in required_columns:
            if column not in table_columns:
                raise DuckDBTableIncorrectColumnsError(
                    self.table_name, required_columns, table_columns
                )

    def _node_to_table_row(self, node: BaseNode) -> Any:
        return (
            node.node_id,
            node.get_content(metadata_mode=MetadataMode.NONE),
            node.get_embedding(),
            node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            ),
        )

    def _table_row_to_node(self, row: Any) -> BaseNode:
        return metadata_dict_to_node(json.loads(row[3]), row[1])

    def _table_row_to_query_result(self, rows: list[Any]) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []

        for row in rows:
            node = self._table_row_to_node(row)
            nodes.append(node)
            ids.append(row[0])
            similarities.append(row[4])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:  # noqa: ARG002
        """Query the vector store for top k most similar nodes."""
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        filter_expression = self._build_metadata_filter_expressions(
            metadata_filters=query.filters
        )

        inner_query = self._table.select(
            StarExpression(),
            FunctionExpression(
                "list_cosine_similarity",
                ColumnExpression("embedding"),
                ConstantExpression(query.query_embedding),
            ).alias("score"),
        ).filter(filter_expression)

        outer_query = (
            inner_query.select(
                ColumnExpression("node_id"),
                ColumnExpression("text"),
                ColumnExpression("embedding"),
                ColumnExpression("metadata_"),
                ColumnExpression("score"),
            )
            .filter(
                ColumnExpression("score").isnotnull(),
            )
            .sort(
                ColumnExpression("score").desc(),
            )
            .limit(
                query.similarity_top_k,
            )
        )

        command = outer_query.sql_query()

        rows = self._conn.execute(command).fetchall()

        return self._table_row_to_query_result(rows)

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:  # noqa: ARG002
        """Add nodes to the vector store."""
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        [self._table.insert(self._node_to_table_row(node)) for node in nodes]

        return [node.node_id for node in nodes]

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **get_kwargs: Any,
    ) -> list[BaseNode]:  # noqa: ARG002
        """Get nodes using node_ids and/or filters. If both are provided, both are considered."""
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        filter_expression = self._build_node_id_metadata_filter_expression(
            node_ids=node_ids,
            filters=filters,
        )

        command = self._table.filter(filter_expression).sql_query()

        rows = self._conn.execute(command).fetchall()

        return [self._table_row_to_node(row) for row in rows]

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:  # noqa: ARG002
        """Delete nodes using node_ids and/or filters. If both are provided, both are considered."""
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        filter_expression = self._build_node_id_metadata_filter_expression(
            node_ids=node_ids,
            filters=filters,
        )

        command = f"DELETE FROM {self.table_name} WHERE {filter_expression}"

        self._conn.execute(command)

    def clear(self, **clear_kwargs: Any) -> None:  # noqa: ARG002
        """Clear the vector store."""
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        command = f"DELETE FROM {self.table_name}"

        self._conn.execute(command)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:  # noqa: ARG002
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        where_clause = self._build_metadata_filter_expression(
            "ref_doc_id", ref_doc_id, FilterOperator.EQ
        )

        command = f"DELETE FROM {self.table_name} WHERE {where_clause}"

        self._conn.execute(command)

    def _build_node_id_metadata_filter_expression(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> Expression:
        filter_expression = Expression(True)

        if filters is not None:
            filter_expression = self._build_metadata_filter_expressions(
                metadata_filters=filters
            )

        if node_ids is not None:
            node_id_expression = FunctionExpression(
                "list_contains",
                ConstantExpression(node_ids),
                ColumnExpression("node_id"),
            )

            filter_expression = filter_expression.__and__(node_id_expression)

        return filter_expression

    def _build_metadata_filter_expression(
        self, key: str, value: Any, operator: FilterOperator
    ) -> Expression:
        metadata_column = ColumnExpression(f"metadata_.{key}")

        sample_value = value[0] if isinstance(value, list) else value
        value_type = filter_value_type_to_duckdb_type.get(type(sample_value))

        metadata_type_expression = FunctionExpression(
            "json_type",
            ColumnExpression("metadata_"),
            ConstantExpression(f"$.{key}"),
        )

        if value_type is None:
            # If the value is a JSON Null, we want to swap the 'Null' for an actual null
            metadata_column = CaseExpression(
                condition=metadata_type_expression.__eq__(ConstantExpression("NULL")),
                value=ConstantExpression(None),
            ).otherwise(metadata_column)

        if value_type == VARCHAR:
            # If the value is a string, it means the column is a JSON string
            # and so we need to unpack it otherwise we'll get back a JSON string (a string wrapped in quotes)
            # https://github.com/duckdb/duckdb/issues/17681
            metadata_column = FunctionExpression(
                "json_extract_string",
                ColumnExpression("metadata_"),
                ConstantExpression(f"$.{key}"),
            )

        metadata_value = ConstantExpression(value)

        return self._build_filter_expression(metadata_column, metadata_value, operator)

    def _build_filter_expression(
        self, column: Expression, value: Expression, operator: FilterOperator
    ) -> Expression:
        # If our operator is IN or NIN we need to use a function expression
        match operator:
            case operator_func if operator_func := li_filter_to_py_operator.get(
                operator
            ):
                # We have a straightforward operator, so we can just use the Python operator
                # i.e. FilterOperator.EQ -> `==` (operator.eq)
                # i.e. FilterOperator.GTE -> `>=` (operator.ge)
                # ...
                return operator_func(column, value)

            # Less straightforward operators
            case FilterOperator.IN:
                # doclument value is in the filter list
                return FunctionExpression(
                    "list_contains",
                    value,
                    column,
                )
            case FilterOperator.NIN:
                # document value is not in the filter list
                return FunctionExpression(
                    "list_contains",
                    value,
                    column,
                ).__eq__(ConstantExpression(False))

            case FilterOperator.CONTAINS:
                # filter_value is in the document value
                # This will never be true so long as the DuckDB vector store
                # requires flat metadata
                return FunctionExpression(
                    "list_contains",
                    value,
                    column,
                )
            case FilterOperator.ANY:
                # array of values has at least one element in common with the array of values in the column
                return FunctionExpression(
                    "list_has_any",
                    column,
                    value,
                )
            case FilterOperator.ALL:
                # array of values is the same as the array of values in the column
                return FunctionExpression(
                    "list_has_all",
                    column,
                    value,
                )
            case FilterOperator.TEXT_MATCH:
                return FunctionExpression(
                    "contains",
                    column,
                    value,
                )
            case FilterOperator.TEXT_MATCH_INSENSITIVE:
                return FunctionExpression(
                    "contains",
                    FunctionExpression(
                        "lower",
                        column,
                    ),
                    FunctionExpression(
                        "lower",
                        value,
                    ),
                )
            case FilterOperator.IS_EMPTY:
                # column is null or the array is empty
                return column.isnull().__or__(
                    CaseExpression(
                        condition=FunctionExpression("typeof", column).__eq__(
                            ConstantExpression("ARRAY")
                        ),
                        value=FunctionExpression("length", column).__eq__(
                            ConstantExpression(0)
                        ),
                    )
                )

            case _:
                raise NotImplementedError(f"Unsupported operator: {operator}")

    def _build_metadata_filter_expressions(
        self, metadata_filters: MetadataFilters | None = None
    ) -> Expression:
        expressions: list[Expression] = []

        if metadata_filters is None or len(metadata_filters.filters) == 0:
            return Expression(True)

        for metadata_filter in metadata_filters.filters:
            if isinstance(metadata_filter, MetadataFilter):
                expressions.append(
                    self._build_metadata_filter_expression(
                        metadata_filter.key,
                        metadata_filter.value,
                        metadata_filter.operator,
                    )
                )
            elif isinstance(metadata_filter, MetadataFilters):
                expressions.append(
                    self._build_metadata_filter_expressions(metadata_filter)
                )
            else:
                raise NotImplementedError(
                    f"Unsupported metadata filter: {metadata_filter}"
                )

        final_expression: Expression = expressions[0]

        for expression in expressions[1:]:
            if metadata_filters.condition == FilterCondition.AND:
                final_expression = final_expression.__and__(expression)
                continue

            if metadata_filters.condition == FilterCondition.OR:
                final_expression = final_expression.__or__(expression)
                continue

            # I dont know what NOT means in this context
            # https://github.com/run-llama/llama_index/discussions/19112
            # if metadata_filters.condition == FilterCondition.NOT:
            #     final_expression = Expression(py_operator.not_(expressions))
            #     continue
            else:
                raise NotImplementedError(
                    f"Unsupported condition: {metadata_filters.condition}"
                )

        return final_expression
