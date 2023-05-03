"""Default query for GPTListIndex."""
import logging
from typing import Any, List, Optional, Tuple

from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.node import Node, NodeWithScore
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.embedding_utils import (
    get_top_k_embeddings,
)
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.list.base import GPTListIndex

logger = logging.getLogger(__name__)


class ListIndexRetriever(BaseRetriever):
    """Simple retriever for ListIndex that returns all nodes.

    Args:
        index (GPTListIndex): The index to retrieve from.

    """

    def __init__(self, index: GPTListIndex, **kwargs: Any) -> None:
        self._index = index

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve nodes."""
        del query_bundle

        node_ids = self._index.index_struct.nodes
        nodes = self._index.docstore.get_nodes(node_ids)
        return [NodeWithScore(node) for node in nodes]


class ListIndexEmbeddingRetriever(BaseRetriever):
    """Embedding based retriever for ListIndex.

    Generates embeddings in a lazy fashion for all
    nodes that are traversed.

    Args:
        index (GPTListIndex): The index to retrieve from.
        similarity_top_k (Optional[int]): The number of top nodes to return.

    """

    def __init__(
        self,
        index: GPTListIndex,
        similarity_top_k: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        self._index = index
        self._similarity_top_k = similarity_top_k

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve nodes."""
        node_ids = self._index.index_struct.nodes
        # top k nodes
        nodes = self._index.docstore.get_nodes(node_ids)
        query_embedding, node_embeddings = self._get_embeddings(query_bundle, nodes)

        top_similarities, top_idxs = get_top_k_embeddings(
            query_embedding,
            node_embeddings,
            similarity_top_k=self._similarity_top_k,
            embedding_ids=list(range(len(nodes))),
        )

        top_k_nodes = [nodes[i] for i in top_idxs]

        node_with_scores = []
        for node, similarity in zip(top_k_nodes, top_similarities):
            node_with_scores.append(NodeWithScore(node, score=similarity))

        logger.debug(f"> Top {len(top_idxs)} nodes:\n")
        nl = "\n"
        logger.debug(f"{ nl.join([n.get_text() for n in top_k_nodes]) }")
        return node_with_scores

    def _get_embeddings(
        self, query_bundle: QueryBundle, nodes: List[Node]
    ) -> Tuple[List[float], List[List[float]]]:
        """Get top nodes by similarity to the query."""
        if query_bundle.embedding is None:
            event_id = self._index._service_context.callback_manager.on_event_start(
                CBEventType.EMBEDDING
            )
            query_bundle.embedding = (
                self._index._service_context.embed_model.get_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )
            self._index._service_context.callback_manager.on_event_end(
                CBEventType.EMBEDDING, payload={"num_nodes": 1}, event_id=event_id
            )
        node_embeddings: List[List[float]] = []

        event_id = self._index._service_context.callback_manager.on_event_start(
            CBEventType.EMBEDDING
        )
        nodes_embedded = 0
        for node in nodes:
            if node.embedding is None:
                nodes_embedded += 1
                node.embedding = (
                    self._index.service_context.embed_model.get_text_embedding(
                        node.get_text()
                    )
                )

            node_embeddings.append(node.embedding)
        self._index._service_context.callback_manager.on_event_end(
            CBEventType.EMBEDDING,
            payload={"num_nodes": nodes_embedded},
            event_id=event_id,
        )
        return query_bundle.embedding, node_embeddings
