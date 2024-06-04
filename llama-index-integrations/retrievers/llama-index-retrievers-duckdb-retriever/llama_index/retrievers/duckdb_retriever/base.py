import logging
from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle

from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.vector_stores.duckdb.base import DuckDBLocalContext

logger = logging.getLogger(__name__)

class DuckDBRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: DuckDBVectorStore,
        text_search_config: Optional[dict] = {
            "stemmer": "english",
            "stopwords": "english",
            "ignore": r"(\\.|[^a-z])+",
            "strip_accents": True,
            "lower": True,
            "overwrite": True,
        },
        # TODO: Add more options for FTS index creation

        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._vector_store = vector_store
        self._similarity_top_k = similarity_top_k
        self._callback_manager = callback_manager
        self._verbose = verbose
        
        # TODO: Check if the vector store already has data

        # Create an FTS index on the 'text' column if it doesn't already exist
        if self._vector_store.database_name==':memory:':
            self._db_path = ':memory:'
        else:
            self._db_path= self._vector_store._database_path

        strip_accents =1 if text_search_config["strip_accents"] else 0
        lower = 1 if text_search_config["lower"] else 0
        overwrite = 1 if text_search_config["overwrite"] else 0
        ignore = text_search_config["ignore"]
        
        sql = f"""
            PRAGMA create_fts_index({vector_store.table_name}, node_id, text, 
                            stemmer = '{text_search_config["stemmer"]}',
                            stopwords = '{text_search_config["stopwords"]}', ignore = '{ignore}',
                            strip_accents = {strip_accents}, lower = {lower}, overwrite = {overwrite})      
                        """
        with DuckDBLocalContext(self._db_path) as conn:
            conn.execute(sql)
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._verbose:
            logger.info(f"Searching for: {query_bundle.query_str}")
        query = query_bundle.query_str
        sql =f"""
                SELECT
                    fts_main_{self._vector_store.table_name}.match_bm25(node_id, '{query}') AS score,
                    node_id, text
                FROM {self._vector_store.table_name}
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT {self._similarity_top_k};
            """
        with DuckDBLocalContext(self._db_path) as conn:
            query_result = conn.execute(sql).fetchall()
        # Convert query result to NodeWithScore objects
        retrieve_nodes = []
        for row in query_result:
            score,node_id, text = row
            node = TextNode(id=node_id, text=text)
            retrieve_nodes.append(NodeWithScore(node=node, score=float(score)))

        return retrieve_nodes