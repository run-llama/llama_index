"""Base vector store index query."""


from typing import Any, List, Optional

from gpt_index.data_structs.data_structs_v2 import IndexDict

# from gpt_index.data_structs.data_structs import IndexDict, Node
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.service_context import ServiceContext
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
        service_context: ServiceContext,
        vector_store: Optional[VectorStore] = None,
        similarity_top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct, service_context=service_context, **kwargs
        )
        self._similarity_top_k = similarity_top_k
        if vector_store is None:
            raise ValueError("Vector store is required for vector store query.")
        self._vector_store = vector_store

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None:
                query_bundle.embedding = (
                    self._service_context.embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
            query_result = self._vector_store.query(
                query_bundle.embedding,
                self._similarity_top_k,
                self._doc_ids,
            )
        else:
            # TODO: fix function signature of query
            query_result = self._vector_store.query(
                [],
                self._similarity_top_k,
                self._doc_ids,
                query_str=query_bundle.query_str,
            )

        if query_result.nodes is None:
            # NOTE: vector store does not keep text and returns node indices.
            # Need to recover all nodes from docstore
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index_struct, IndexDict)
            node_ids = [self._index_struct.nodes_dict[idx] for idx in query_result.ids]
            nodes = self._docstore.get_nodes(node_ids)
            query_result.nodes = nodes
        else:
            # NOTE: vector store keeps text, returns nodes.
            # Only need to recover image or index nodes from docstore
            for i in range(len(query_result.nodes)):
                node_id = query_result.nodes[i].get_doc_id()
                if node_id in self._docstore.docs:
                    query_result.nodes[i] = self._docstore.get_node(node_id)

        log_vector_store_query_result(query_result)

        if similarity_tracker is not None and query_result.similarities is not None:
            for node, similarity in zip(query_result.nodes, query_result.similarities):
                similarity_tracker.add(node, similarity)

        return query_result.nodes
