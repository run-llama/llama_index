"""DuckDB vector store."""

import json
import logging
import operator
from pathlib import Path
from typing import Any, cast

import duckdb
from duckdb import (
    ColumnExpression,
    ConstantExpression,
    DuckDBPyRelation,
    FunctionExpression,
    StarExpression,
)
from duckdb.typing import FLOAT, INTEGER, VARCHAR
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

op_string_to_op = {
    FilterOperator.GT: operator.gt,
    FilterOperator.GTE: operator.ge,
    FilterOperator.LT: operator.lt,
    FilterOperator.LTE: operator.le,
    FilterOperator.EQ: operator.eq,
    FilterOperator.NE: operator.ne,
}

filter_value_type_to_duckdb_type = {
    StrictInt: INTEGER,
    StrictFloat: FLOAT,
    StrictStr: VARCHAR,
    int: INTEGER,
    float: FLOAT,
    str: VARCHAR,
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

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:  # noqa: ARG002
        """
        Query index for top k most similar nodes.
        """
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        inner_query = self._table

        if query.filters is not None:
            inner_query = self._build_metadata_filter_relation(
                inner_query, query.filters
            )

        inner_query = inner_query.select(
            StarExpression(),
            FunctionExpression(
                "list_cosine_similarity",
                ColumnExpression("embedding"),
                ConstantExpression(query.query_embedding),
            ).alias("score"),
        )

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

        nodes = []
        similarities = []
        ids = []

        for row in rows:
            node = self._table_row_to_node(row)
            nodes.append(node)
            ids.append(row[0])
            similarities.append(row[4])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:  # noqa: ARG002
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
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
        """
        Get nodes using node_ids or filters.
        """
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        rows_to_get = self._table

        if node_ids is not None:
            rows_to_get = rows_to_get.filter(
                FunctionExpression(
                    "list_contains",
                    ConstantExpression(node_ids),
                    ColumnExpression("node_id"),
                )
            )
        if filters is not None:
            rows_to_get = self._build_metadata_filter_relation(rows_to_get, filters)

        command = rows_to_get.sql_query()

        rows = self._conn.execute(command).fetchall()

        return [self._table_row_to_node(row) for row in rows]

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:  # noqa: ARG002
        """
        Delete nodes using node_ids or filters.
        """
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        rows_to_delete = self._table

        if node_ids is not None:
            rows_to_delete = rows_to_delete.filter(
                FunctionExpression(
                    "list_contains",
                    ConstantExpression(node_ids),
                    ColumnExpression("node_id"),
                )
            )
        if filters is not None:
            rows_to_delete = self._build_metadata_filter_relation(
                rows_to_delete, filters
            )

        rows_to_delete = rows_to_delete.select(ColumnExpression("node_id"))

        self._conn.execute(
            f"DELETE FROM {self.table_name} WHERE node_id IN ({rows_to_delete.sql_query()})"
        )

    def clear(self, **clear_kwargs: Any) -> None:  # noqa: ARG002
        """Clear the vector store."""
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        self._conn.execute(f"DELETE FROM {self.table_name}")  # noqa: S608

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:  # noqa: ARG002
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        self._conn.execute(
            f"DELETE FROM {self.table_name} WHERE metadata_.ref_doc_id::string = $doc_id",  # noqa: S608
            parameters={"doc_id": ref_doc_id},
        )

    def _build_metadata_filter_relation(
        self, relation: DuckDBPyRelation, standard_filters: MetadataFilters
    ) -> DuckDBPyRelation:
        if self._table is None:
            raise DuckDBTableNotInitializedError(self.table_name)

        new_relation = relation

        for metadata_filter in standard_filters.filters:
            metadata_filter = cast("MetadataFilter", metadata_filter)

            operator = op_string_to_op[metadata_filter.operator]

            value_type = filter_value_type_to_duckdb_type[type(metadata_filter.value)]

            column_name = f"metadata_.{metadata_filter.key}"

            new_relation = new_relation.filter(
                operator(
                    FunctionExpression(
                        "json_extract_string",
                        ColumnExpression(column_name),
                        ConstantExpression("$"),
                    ),
                    ConstantExpression(metadata_filter.value),
                )
            )

        return new_relation
