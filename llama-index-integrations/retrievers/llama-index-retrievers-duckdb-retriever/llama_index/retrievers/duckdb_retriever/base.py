import logging
import os
from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)
import_err_msg = "`duckdb` package not found, please run `pip install duckdb`"


class DuckDBLocalContext:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self._conn = None
        self._home_dir = os.path.expanduser("~")

    def __enter__(self) -> "duckdb.DuckDBPyConnection":
        try:
            import duckdb
        except ImportError:
            raise ImportError(import_err_msg)

        if not os.path.exists(os.path.dirname(self.database_path)):
            raise ValueError(
                f"Directory {os.path.dirname(self.database_path)} does not exist."
            )

        # if not os.path.isfile(self.database_path):
        #     raise ValueError(f"Database path {self.database_path} is not a valid file.")

        self._conn = duckdb.connect(self.database_path)
        self._conn.execute(f"SET home_directory='{self._home_dir}';")

        self._conn.install_extension("fts")
        self._conn.load_extension("fts")

        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._conn.close()

        if self._conn:
            self._conn.close()


class DuckDBRetriever(BaseRetriever):
    def __init__(
        self,
        database_name: Optional[str] = ":memory:",
        table_name: Optional[str] = "documents",
        text_search_config: Optional[dict] = {
            "stemmer": "english",
            "stopwords": "english",
            "ignore": r"(\\.|[^a-z])+",
            "strip_accents": True,
            "lower": True,
            "overwrite": True,
        },
        persist_dir: Optional[str] = "./storage",
        node_id_column: Optional[str] = "node_id",
        text_column: Optional[str] = "text",
        # TODO: Add more options for FTS index creation
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._similarity_top_k = similarity_top_k
        self._callback_manager = callback_manager
        self._verbose = verbose
        self._table_name = table_name
        self._node_id_column = node_id_column
        self._text_column = text_column

        # TODO: Check if the vector store already has data

        # Create an FTS index on the 'text' column if it doesn't already exist
        if database_name == ":memory:":
            self._database_path = ":memory:"
        else:
            self._database_path = os.path.join(persist_dir, database_name)

        strip_accents = 1 if text_search_config["strip_accents"] else 0
        lower = 1 if text_search_config["lower"] else 0
        overwrite = 1 if text_search_config["overwrite"] else 0
        ignore = text_search_config["ignore"]

        sql = f"""
            PRAGMA create_fts_index({self._table_name}, {self._node_id_column}, {self._text_column},
                            stemmer = '{text_search_config["stemmer"]}',
                            stopwords = '{text_search_config["stopwords"]}', ignore = '{ignore}',
                            strip_accents = {strip_accents}, lower = {lower}, overwrite = {overwrite})
                        """
        with DuckDBLocalContext(self._database_path) as conn:
            conn.execute(sql)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._verbose:
            logger.info(f"Searching for: {query_bundle.query_str}")
        query = query_bundle.query_str
        sql = f"""
                SELECT
                    fts_main_{self._table_name}.match_bm25({self._node_id_column}, '{query}') AS score,
                    {self._node_id_column}, {self._text_column}
                FROM {self._table_name}
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT {self._similarity_top_k};
            """
        with DuckDBLocalContext(self._database_path) as conn:
            query_result = conn.execute(sql).fetchall()
        # Convert query result to NodeWithScore objects
        retrieve_nodes = []
        for row in query_result:
            score, node_id, text = row
            node = TextNode(id=node_id, text=text)
            retrieve_nodes.append(NodeWithScore(node=node, score=float(score)))

        return retrieve_nodes
