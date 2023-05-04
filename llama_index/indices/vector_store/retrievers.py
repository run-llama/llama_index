"""Base vector store index query."""


from typing import Any, List, Optional
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K

from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.data_structs import IndexDict
from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.utils import log_vector_store_query_result
from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)


class VectorIndexRetriever(BaseRetriever):
    """Base vector store query.

    Args:
        embed_model (Optional[BaseEmbedding]): embedding model
        similarity_top_k (int): number of top k results to return
        vector_store (Optional[VectorStore]): vector store

    """

    def __init__(
        self,
        index: GPTVectorStoreIndex,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: str = VectorStoreQueryMode.DEFAULT,
        alpha: Optional[float] = None,
        doc_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._vector_store = self._index.vector_store
        self._service_context = self._index.service_context
        self._docstore = self._index.docstore

        self._similarity_top_k = similarity_top_k
        self._vector_store_query_mode = VectorStoreQueryMode(vector_store_query_mode)
        self._alpha = alpha
        self._doc_ids = doc_ids

    @llm_token_counter("retrieve")
    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None:
                event_id = self._service_context.callback_manager.on_event_start(
                    CBEventType.EMBEDDING
                )
                query_bundle.embedding = (
                    self._service_context.embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
                self._service_context.callback_manager.on_event_end(
                    CBEventType.EMBEDDING, payload={"num_nodes": 1}, event_id=event_id
                )

        query = VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
            doc_ids=self._doc_ids,
            query_str=query_bundle.query_str,
            mode=self._vector_store_query_mode,
            alpha=self._alpha,
        )
        query_result = self._vector_store.query(query)

        if query_result.nodes is None:
            # NOTE: vector store does not keep text and returns node indices.
            # Need to recover all nodes from docstore
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index.index_struct, IndexDict)
            node_ids = [
                self._index.index_struct.nodes_dict[idx] for idx in query_result.ids
            ]
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

        node_with_scores: List[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            node_with_scores.append(NodeWithScore(node, score=score))

        return node_with_scores
