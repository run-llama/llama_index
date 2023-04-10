"""Embedding query for list index."""
import logging
from typing import Any, List, Optional, Tuple

from gpt_index.data_structs.data_structs_v2 import IndexList
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.list.query import BaseGPTListIndexQuery
from gpt_index.indices.query.embedding_utils import (
    SimilarityTracker,
    get_top_k_embeddings,
)
from gpt_index.indices.query.schema import QueryBundle

logger = logging.getLogger(__name__)


class GPTListIndexEmbeddingQuery(BaseGPTListIndexQuery):
    """GPTListIndex query.

    An embedding-based query for GPTListIndex, which traverses
    each node in sequence and retrieves top-k nodes by
    embedding similarity to the query.
    Set when `mode="embedding"` in `query` method of `GPTListIndex`.

    .. code-block:: python

        response = index.query("<query_str>", mode="embedding")

    See BaseGPTListIndexQuery for arguments.

    """

    def __init__(
        self,
        index_struct: IndexList,
        similarity_top_k: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct,
            **kwargs,
        )
        self.similarity_top_k = similarity_top_k

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        node_ids = self.index_struct.nodes
        # top k nodes
        nodes = self._docstore.get_nodes(node_ids)
        query_embedding, node_embeddings = self._get_embeddings(query_bundle, nodes)

        top_similarities, top_idxs = get_top_k_embeddings(
            query_embedding,
            node_embeddings,
            similarity_top_k=self.similarity_top_k,
            embedding_ids=list(range(len(nodes))),
        )

        top_k_nodes = [nodes[i] for i in top_idxs]

        if similarity_tracker is not None:
            for node, similarity in zip(top_k_nodes, top_similarities):
                similarity_tracker.add(node, similarity)

        logger.debug(f"> Top {len(top_idxs)} nodes:\n")
        nl = "\n"
        logger.debug(f"{ nl.join([n.get_text() for n in top_k_nodes]) }")
        return top_k_nodes

    def _get_embeddings(
        self, query_bundle: QueryBundle, nodes: List[Node]
    ) -> Tuple[List[float], List[List[float]]]:
        """Get top nodes by similarity to the query."""
        if query_bundle.embedding is None:
            query_bundle.embedding = (
                self._service_context.embed_model.get_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )
        node_embeddings: List[List[float]] = []
        for node in nodes:
            if node.embedding is None:
                node.embedding = self._service_context.embed_model.get_text_embedding(
                    node.get_text()
                )

            node_embeddings.append(node.embedding)
        return query_bundle.embedding, node_embeddings
