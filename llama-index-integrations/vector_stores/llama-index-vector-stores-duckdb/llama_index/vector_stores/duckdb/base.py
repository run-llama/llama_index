"""DuckDB vector store."""

import asyncio
import json
import logging
import operator as py_operator
import threading
from pathlib import Path
from typing import Any, Optional
from typing_extensions import override

import duckdb
from duckdb import (
    ColumnExpression,
    ConstantExpression,
    FunctionExpression,
    StarExpression,
    CaseExpression,
    Expression,
)
from duckdb.typing import FLOAT, INTEGER, VARCHAR, SQLNULL
from collections.abc import Sequence
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
import pyarrow

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
    embed_dim: Optional[int]
    # hybrid_search: Optional[bool] # TODO: support hybrid search
    text_search_config: Optional[dict]
    persist_dir: str

    _shared_conn: Optional[duckdb.DuckDBPyConnection] = PrivateAttr(default=None)
    _thread_local: threading.local = PrivateAttr(default_factory=threading.local)

    _is_initialized: bool = PrivateAttr(default=False)
    _database_path: Optional[str] = PrivateAttr()

    def __init__(
        self,
        database_name: str = ":memory:",
        table_name: str = "documents",
        embed_dim: Optional[int] = None,
        # https://duckdb.org/docs/extensions/full_text_search
        text_search_config: Optional[dict] = None,
        persist_dir: str = "./storage",
        client: Optional[duckdb.DuckDBPyConnection] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Init params."""
        if text_search_config is None:
            text_search_config = DEFAULT_TEXT_SEARCH_CONFIG

        fields = {
            "database_name": database_name,
            "table_name": table_name,
            "embed_dim": embed_dim,
            "text_search_config": text_search_config,
            "persist_dir": persist_dir,
        }

        if client is not None:
            self._shared_conn = client

        super().__init__(stores_text=True, **fields)

        _ = self._initialize_table(self.client, self.table_name, self.embed_dim)

    @classmethod
    def from_local(
        cls,
        database_path: str,
        table_name: str = "documents",
        # schema_name: Optional[str] = "main",
        embed_dim: Optional[int] = None,
        # hybrid_search: Optional[bool] = False,
        text_search_config: Optional[dict] = None,
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
        embed_dim: Optional[int] = None,
        # hybrid_search: Optional[bool] = False,
        text_search_config: Optional[dict] = None,
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
    def client(self) -> duckdb.DuckDBPyConnection:
        """Return client."""
        if self._shared_conn is None:
            self._shared_conn = self._connect(self.database_name, self.persist_dir)

        if not hasattr(self._thread_local, "conn") or self._thread_local.conn is None:
            self._thread_local.conn = self._shared_conn.cursor()

        return self._thread_local.conn

    @classmethod
    def _connect(
        cls, database_name: str, persist_dir: str
    ) -> duckdb.DuckDBPyConnection:
        """Connect to the DuckDB database -- create the data persistence directory if it doesn't exist."""
        database_connection = database_name

        if database_name != ":memory:":
            persist_path = Path(persist_dir)

            if not persist_path.exists():
                persist_path.mkdir(parents=True, exist_ok=True)

            database_connection = str(persist_path / database_name)

        return duckdb.connect(database_connection)

    @property
    def table(self) -> duckdb.DuckDBPyRelation:
        """Return the table for the connection to the DuckDB database."""
        return self.client.table(self.table_name)

    @classmethod
    def _get_embedding_type(cls, embed_dim: Optional[int]) -> str:
        return f"FLOAT[{embed_dim}]" if embed_dim is not None else "FLOAT[]"

    @classmethod
    def _initialize_table(
        cls, conn: duckdb.DuckDBPyConnection, table_name: str, embed_dim: Optional[int]
    ) -> None:
        """Initialize the DuckDB Database, extensions, and documents table."""
        home_dir = Path.home()
        conn.execute(f"SET home_directory='{home_dir}';")
        conn.install_extension("json")
        conn.load_extension("json")
        conn.install_extension("fts")
        conn.load_extension("fts")

        embedding_type = cls._get_embedding_type(embed_dim)

        conn.begin().execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name}  (
                node_id VARCHAR PRIMARY KEY,
                text TEXT,
                embedding {embedding_type},
                metadata_ JSON
            );
        """).commit()

        table = conn.table(table_name)

        required_columns = ["node_id", "text", "embedding", "metadata_"]
        table_columns = table.describe().columns

        for column in required_columns:
            if column not in table_columns:
                raise DuckDBTableIncorrectColumnsError(
                    table_name, required_columns, table_columns
                )

    def _node_to_arrow_row(self, node: BaseNode) -> dict:
        return {
            "node_id": node.node_id,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "embedding": node.get_embedding(),
            "metadata_": node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            ),
        }

    def _arrow_row_to_node(self, row_dict: dict) -> BaseNode:
        node = metadata_dict_to_node(
            metadata=json.loads(row_dict["metadata_"]), text=row_dict["text"]
        )
        node.embedding = row_dict["embedding"]

        return node

    def _arrow_row_to_query_result(self, rows: list[dict]) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []

        for row in rows:
            node = self._arrow_row_to_node(row)
            nodes.append(node)
            ids.append(row["node_id"])
            similarities.append(row["score"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    @override
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:  # noqa: ARG002
        """Query the vector store for top k most similar nodes."""
        filter_expression = self._build_metadata_filter_expressions(
            metadata_filters=query.filters
        )

        inner_query = self.table.select(
            StarExpression(),
            FunctionExpression(
                "array_cosine_similarity"
                if self.embed_dim is not None
                else "list_cosine_similarity",
                ColumnExpression("embedding"),
                ConstantExpression(query.query_embedding).cast(
                    self._get_embedding_type(self.embed_dim)
                ),
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

        rows = self.client.execute(command).arrow().to_pylist()

        return self._arrow_row_to_query_result(rows)

    @override
    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:  # noqa: ARG002
        """Query the vector store for top k most similar nodes."""
        return await asyncio.to_thread(self.query, query, **kwargs)

    @override
    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:  # noqa: ARG002
        """Add nodes to the vector store."""
        rows: list[dict[str, Any]] = [self._node_to_arrow_row(node) for node in nodes]

        arrow_table = pyarrow.Table.from_pylist(rows)
        self.client.from_arrow(arrow_table).insert_into(self.table.alias)
        return [node.node_id for node in nodes]

    @override
    async def async_add(
        self, nodes: Sequence[BaseNode], **add_kwargs: Any
    ) -> list[str]:  # noqa: ARG002
        """Add nodes to the vector store."""
        return await asyncio.to_thread(self.add, nodes, **add_kwargs)

    @override
    def get_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **get_kwargs: Any,
    ) -> list[BaseNode]:  # noqa: ARG002
        """Get nodes using node_ids and/or filters. If both are provided, both are considered."""
        filter_expression = self._build_node_id_metadata_filter_expression(
            node_ids=node_ids,
            filters=filters,
        )

        command = self.table.filter(filter_expression).sql_query()

        rows = self.client.execute(command).arrow().to_pylist()

        return [self._arrow_row_to_node(row) for row in rows]

    @override
    async def aget_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **get_kwargs: Any,
    ) -> list[BaseNode]:  # noqa: ARG002
        """Get nodes using node_ids and/or filters. If both are provided, both are considered."""
        return await asyncio.to_thread(self.get_nodes, node_ids, filters, **get_kwargs)

    @override
    def delete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:  # noqa: ARG002
        """Delete nodes using node_ids and/or filters. If both are provided, both are considered."""
        filter_expression = self._build_node_id_metadata_filter_expression(
            node_ids=node_ids,
            filters=filters,
        )

        command = f"DELETE FROM {self.table.alias} WHERE {filter_expression}"

        self.client.execute(command)

    @override
    async def adelete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:  # noqa: ARG002
        """Delete nodes using node_ids and/or filters. If both are provided, both are considered."""
        return await asyncio.to_thread(
            self.delete_nodes, node_ids, filters, **delete_kwargs
        )

    @override
    def clear(self, **clear_kwargs: Any) -> None:  # noqa: ARG002
        """Clear the vector store."""
        command = f"DELETE FROM {self.table.alias}"

        self.client.execute(command)

    @override
    async def aclear(self, **clear_kwargs: Any) -> None:  # noqa: ARG002
        """Clear the vector store."""
        return await asyncio.to_thread(self.clear, **clear_kwargs)

    @override
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:  # noqa: ARG002
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        where_clause = self._build_metadata_filter_expression(
            "ref_doc_id", ref_doc_id, FilterOperator.EQ
        )

        command = f"DELETE FROM {self.table.alias} WHERE {where_clause}"

        self.client.execute(command)

    @override
    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:  # noqa: ARG002
        """
        Delete nodes using with ref_doc_id.

        """
        return await asyncio.to_thread(self.delete, ref_doc_id, **delete_kwargs)

    def _build_node_id_metadata_filter_expression(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
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
        """
        Build a filter expression for a given column and value.

        Args:
            column: The key in the document to use in the filter.
            value: The value to use in the filter.
            operator: The filter operator to use.

        """
        if operator_func := li_filter_to_py_operator.get(operator):
            # We have a straightforward operator, and DuckDB can handle just take the Python operator
            # i.e. FilterOperator.EQ -> `==` (operator.eq)
            # i.e. FilterOperator.GTE -> `>=` (operator.ge)
            # ...
            return operator_func(column, value)

        if operator == FilterOperator.IN:
            # Given a list of values, check to see if the document's value is in the list
            return FunctionExpression(
                "list_contains",  # list_contains(list_to_look_in, element_to_find)
                value,
                column,
            )

        if operator == FilterOperator.NIN:
            # Given a list of values, check to see if the document's value is not in the list
            return FunctionExpression(
                "list_contains",  # list_contains(list_to_look_in, element_to_find)
                value,
                column,
            ).__eq__(ConstantExpression(False))

        if operator == FilterOperator.CONTAINS:
            # filter_value is in the document value
            # This will never be true so long as the DuckDB vector store
            # requires flat metadata
            return Expression(False)
            # return FunctionExpression(
            #     "list_contains", # list_contains(list_to_look_in, element_to_find)
            #     value,
            #     column,
            # )
        if operator == FilterOperator.ANY:
            # Check if the intersection of the two lists has at least one element
            return FunctionExpression(
                "list_has_any",
                column,
                value,
            )

        if operator == FilterOperator.ALL:
            # Check if all of the provided values are in the document's value
            return FunctionExpression(
                "list_has_all",  # list_has_all(list, sub-list)
                column,
                value,
            )

        if operator == FilterOperator.TEXT_MATCH:
            return FunctionExpression(
                "contains",
                column,
                value,
            )

        if operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
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

        if operator == FilterOperator.IS_EMPTY:
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

        raise NotImplementedError(f"Unsupported operator: {operator}")

    def _build_metadata_filter_expressions(
        self, metadata_filters: Optional[MetadataFilters] = None
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
            # We will do an implicit AND for NOT conditions
            if metadata_filters.condition in [FilterCondition.AND, FilterCondition.NOT]:
                final_expression = final_expression.__and__(expression)
                continue

            if metadata_filters.condition == FilterCondition.OR:
                final_expression = final_expression.__or__(expression)
                continue

            raise NotImplementedError(
                f"Unsupported condition: {metadata_filters.condition}"
            )

        if metadata_filters.condition == FilterCondition.NOT:
            final_expression = final_expression.__invert__()

        return final_expression
