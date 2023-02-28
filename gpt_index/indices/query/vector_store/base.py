"""Base vector store index query."""


from typing import Any, List, Optional

from gpt_index.data_structs.data_structs import IndexDict, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.utils import log_vector_store_query_result
from gpt_index.vector_stores.types import VectorStore


class GPTVectorStoreIndexQuery(BaseGPTIndexQuery[IndexDict]):
    """Base vector store query.

    Args:
        embed_model (Optional[BaseEmbedding]): embedding model
        similarity_top_k (int): number of top k results to return
        vector_store (Optional[VectorStore]): vector store

    """

    def __init__(
        self,
        index_struct: IndexDict,
        vector_store: Optional[VectorStore] = None,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, embed_model=embed_model, **kwargs)
        self._similarity_top_k = similarity_top_k
        if vector_store is None:
            raise ValueError("Vector store is required for vector store query.")
        self._vector_store = vector_store

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        query_embedding = self._embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )

        query_result = self._vector_store.query(query_embedding, self._similarity_top_k)

        if query_result.nodes is None:
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index_struct, IndexDict)
            nodes = self._index_struct.get_nodes(query_result.ids)
            query_result.nodes = nodes

        log_vector_store_query_result(query_result)

        if similarity_tracker is not None and query_result.similarities is not None:
            for node, similarity in zip(query_result.nodes, query_result.similarities):
                similarity_tracker.add(node, similarity)

        return query_result.nodes
