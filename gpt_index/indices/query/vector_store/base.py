"""Base vector store index query."""


from typing import Any, List, Optional

from gpt_index.data_structs.data_structs import IndexDict, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.utils import log_vector_store_query_result
from gpt_index.vector_stores.simple import SimpleVectorStore
from gpt_index.vector_stores.types import VectorStore


class GPTVectorStoreIndexQuery(BaseGPTIndexQuery[IndexDict]):
    """Base vector store query."""

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

        # TODO: this is a temporary hack to allow composable
        # indices to work for simple vector stores
        # Our composability framework at the moment only allows for storage
        # of index_struct, not vector_store. Therefore in order to
        # allow simple vector indices to be composed, we need to "infer"
        # the vector store from the index struct.
        # NOTE: the next refactor would be to allow users to pass in
        # the vector store during query-time. However this is currently
        # not complete in our composability framework because the configs
        # are keyed on index type, not index id (which means that users
        # can't pass in distinct vector stores for different subindices).
        # NOTE: composability on top of other vector stores (pinecone/weaviate)
        # was already broken in this form.
        if vector_store is None:
            if len(index_struct.embeddings_dict) > 0:
                simple_vector_store_data_dict = {
                    "embedding_dict": index_struct.embeddings_dict,
                }
                vector_store = SimpleVectorStore(
                    simple_vector_store_data_dict=simple_vector_store_data_dict
                )
            else:
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
