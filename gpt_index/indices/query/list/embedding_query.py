"""Embedding query for list index."""
import logging
from typing import Any, List, Optional, Tuple

from gpt_index.data_structs.data_structs import IndexList, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.embedding_utils import (
    SimilarityTracker,
    get_top_k_embeddings,
)
from gpt_index.indices.query.list.query import BaseGPTListIndexQuery
from gpt_index.indices.query.schema import QueryBundle


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
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct,
            embed_model=embed_model,
            **kwargs,
        )
        self.similarity_top_k = similarity_top_k

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        nodes = self.index_struct.nodes
        # top k nodes
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

        logging.debug(f"> Top {len(top_idxs)} nodes:\n")
        nl = "\n"
        logging.debug(f"{ nl.join([n.get_text() for n in top_k_nodes]) }")
        return top_k_nodes

    def _get_embeddings(
        self, query_bundle: QueryBundle, nodes: List[Node]
    ) -> Tuple[List[float], List[List[float]]]:
        """Get top nodes by similarity to the query."""
        query_embedding = self._embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )
        node_embeddings: List[List[float]] = []
        for node in self.index_struct.nodes:
            if node.embedding is not None:
                text_embedding = node.embedding
            else:
                text_embedding = self._embed_model.get_text_embedding(node.get_text())
                node.embedding = text_embedding

            node_embeddings.append(text_embedding)
        return query_embedding, node_embeddings
