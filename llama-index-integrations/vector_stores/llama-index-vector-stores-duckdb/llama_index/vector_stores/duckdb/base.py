"""DuckDB vector store."""

import json
import logging
import os
from typing import Any, List, Optional, cast

from fsspec.utils import Sequence
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

import duckdb

logger = logging.getLogger(__name__)
import_err_msg = "`duckdb` package not found, please run `pip install duckdb`"


class DuckDBLocalContext:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self._conn = None
        self._home_dir = os.path.expanduser("~")

    def __enter__(self) -> "duckdb.DuckDBPyConnection":
        if not os.path.exists(os.path.dirname(self.database_path)):
            raise ValueError(
                f"Directory {os.path.dirname(self.database_path)} does not exist."
            )

        # if not os.path.isfile(self.database_path):
        #     raise ValueError(f"Database path {self.database_path} is not a valid file.")

        self._conn = duckdb.connect(self.database_path)
        self._conn.execute(f"SET home_directory='{self._home_dir}';")
        self._conn.install_extension("json")
        self._conn.load_extension("json")
        self._conn.install_extension("fts")
        self._conn.load_extension("fts")

        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn:
            self._conn.close()


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

    database_name: Optional[str]
    table_name: str
    # schema_name: Optional[str] # TODO: support schema name
    embed_dim: Optional[int]
    # hybrid_search: Optional[bool] # TODO: support hybrid search
    text_search_config: Optional[dict]
    persist_dir: Optional[str]

    _conn: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)
    _database_path: Optional[str] = PrivateAttr()

    def __init__(
        self,
        database_name: str = ":memory:",
        table_name: str = "documents",
        embed_dim: Optional[int] = None,
        # https://duckdb.org/docs/extensions/full_text_search
        text_search_config: Optional[dict] = {
            "stemmer": "english",
            "stopwords": "english",
            "ignore": "(\\.|[^a-z])+",
            "strip_accents": True,
            "lower": True,
            "overwrite": False,
        },
        persist_dir: str = "./storage",
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        database_path = None
        if database_name == ":memory:":
            _home_dir = os.path.expanduser("~")
            conn = duckdb.connect(database_name)
            conn.execute(f"SET home_directory='{_home_dir}';")
            conn.install_extension("json")
            conn.load_extension("json")
            conn.install_extension("fts")
            conn.load_extension("fts")
        else:
            # check if persist dir exists
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)

            database_path = os.path.join(persist_dir, database_name)

            with DuckDBLocalContext(database_path) as _conn:
                pass

            conn = None

        fields = {
            "database_name": database_name,
            "table_name": table_name,
            "embed_dim": embed_dim,
            "text_search_config": text_search_config,
            "persist_dir": persist_dir,
        }
        super().__init__(stores_text=True, **fields)
        self._is_initialized = False
        self._conn = conn
        self._database_path = database_path

    @classmethod
    def from_local(
        cls,
        database_path: str,
        table_name: str = "documents",
        # schema_name: Optional[str] = "main",
        embed_dim: Optional[int] = None,
        # hybrid_search: Optional[bool] = False,
        text_search_config: Optional[dict] = {
            "stemmer": "english",
            "stopwords": "english",
            "ignore": "(\\.|[^a-z])+",
            "strip_accents": True,
            "lower": True,
            "overwrite": False,
        },
        **kwargs: Any,
    ) -> "DuckDBVectorStore":
        """Load a DuckDB vector store from a local file."""
        with DuckDBLocalContext(database_path) as _conn:
            try:
                _table_info = _conn.execute(f"SHOW {table_name};").fetchall()
            except Exception as e:
                raise ValueError(f"Index table {table_name} not found in the database.")

            # Not testing for the column type similarity only testing for the column names.
            _std = {"text", "node_id", "embedding", "metadata_"}
            _ti = {_i[0] for _i in _table_info}
            if _std != _ti:
                raise ValueError(
                    f"Index table {table_name} does not have the correct schema."
                )

        _cls = cls(
            database_name=os.path.basename(database_path),
            table_name=table_name,
            embed_dim=embed_dim,
            text_search_config=text_search_config,
            persist_dir=os.path.dirname(database_path),
            **kwargs,
        )
        _cls._is_initialized = True

        return _cls

    @classmethod
    def from_params(
        cls,
        database_name: str = ":memory:",
        table_name: str = "documents",
        # schema_name: Optional[str] = "main",
        embed_dim: Optional[int] = None,
        # hybrid_search: Optional[bool] = False,
        text_search_config: Optional[dict] = {
            "stemmer": "english",
            "stopwords": "english",
            "ignore": "(\\.|[^a-z])+",
            "strip_accents": True,
            "lower": True,
            "overwrite": False,
        },
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

    def _initialize(self) -> None:
        if not self._is_initialized:
            # TODO: schema.table also.
            # Check if table and type is present
            # if not, create table
            if self.embed_dim is None:
                _query = f"""
                    CREATE TABLE {self.table_name} (
                        node_id VARCHAR,
                        text TEXT,
                        embedding FLOAT[],
                        metadata_ JSON
                        );
                    """
            else:
                _query = f"""
                    CREATE TABLE {self.table_name} (
                        node_id VARCHAR,
                        text TEXT,
                        embedding FLOAT[{self.embed_dim}],
                        metadata_ JSON
                        );
                    """

            if self.database_name == ":memory:":
                self._conn.execute(_query)
            elif self._database_path is not None:
                with DuckDBLocalContext(self._database_path) as _conn:
                    _conn.execute(_query)

            self._is_initialized = True

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

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        self._initialize()

        ids = []

        if self.database_name == ":memory:":
            _table = self._conn.table(self.table_name)
            for node in nodes:
                ids.append(node.node_id)
                _row = self._node_to_table_row(node)
                _table.insert(_row)
        elif self._database_path is not None:
            with DuckDBLocalContext(self._database_path) as _conn:
                _table = _conn.table(self.table_name)
                for node in nodes:
                    ids.append(node.node_id)
                    _row = self._node_to_table_row(node)
                    _table.insert(_row)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        _ddb_query = f"""
            DELETE FROM {self.table_name}
            WHERE json_extract_string(metadata_, '$.ref_doc_id') = ?;
            """
        if self.database_name == ":memory:":
            self._conn.execute(_ddb_query, [ref_doc_id])
        elif self._database_path is not None:
            with DuckDBLocalContext(self._database_path) as _conn:
                _conn.execute(_ddb_query, [ref_doc_id])

    @staticmethod
    def _build_metadata_filter_condition(
        standard_filters: MetadataFilters,
    ) -> str:
        """Translate standard metadata filters to DuckDB SQL specification."""
        filters_list = []
        # condition = standard_filters.condition or "and"  ## and/or as strings.
        condition = "AND"
        _filters_condition_list = []

        for filter in standard_filters.filters:
            filter = cast(MetadataFilter, filter)
            if filter.operator:
                if filter.operator in [
                    "<",
                    ">",
                    "<=",
                    ">=",
                    "<>",
                    "!=",
                ]:
                    filters_list.append((filter.key, filter.operator, filter.value))
                elif filter.operator in ["=="]:
                    filters_list.append((filter.key, "=", filter.value))
                else:
                    raise ValueError(
                        f"Filter operator {filter.operator} not supported."
                    )
            else:
                filters_list.append((filter.key, "=", filter.value))

        for _fc in filters_list:
            if isinstance(_fc[2], str):
                _filters_condition_list.append(
                    f"json_extract_string(metadata_, '$.{_fc[0]}') {_fc[1]} '{_fc[2]}'"
                )
            else:
                _filters_condition_list.append(
                    f"json_extract(metadata_, '$.{_fc[0]}') {_fc[1]} {_fc[2]}"
                )

        return f" {condition} ".join(_filters_condition_list)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query.query_embedding (List[float]): query embedding
            query.similarity_top_k (int): top k most similar nodes

        """
        nodes = []
        similarities = []
        ids = []

        if query.filters is not None:
            # TODO: results from the metadata filter query
            _filter_string = self._build_metadata_filter_condition(query.filters)
            _ddb_query = f"""
            SELECT node_id, text, embedding, metadata_, score
            FROM (
                SELECT *, list_cosine_similarity(embedding, ?) AS score
                FROM {self.table_name}
                WHERE ?
            ) sq
            WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT ?;
            """
            query_params = [
                query.query_embedding,
                _filter_string,
                query.similarity_top_k,
            ]
        else:
            _ddb_query = f"""
            SELECT node_id, text, embedding, metadata_, score
            FROM (
                SELECT *, list_cosine_similarity(embedding, ?) AS score
                FROM {self.table_name}
            ) sq
            WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT ?;
            """
            query_params = [
                query.query_embedding,
                query.similarity_top_k,
            ]

        _final_results = []
        if self.database_name == ":memory:":
            _final_results = self._conn.execute(_ddb_query, query_params).fetchall()
        elif self._database_path is not None:
            with DuckDBLocalContext(self._database_path) as _conn:
                _final_results = _conn.execute(_ddb_query, query_params).fetchall()

        for _row in _final_results:
            node = self._table_row_to_node(_row)
            nodes.append(node)
            similarities.append(_row[4])
            ids.append(_row[0])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
