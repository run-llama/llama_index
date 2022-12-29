"""Embedding query for list index."""
from typing import Any, List, Optional, Tuple

from gpt_index.data_structs.data_structs import IndexList, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.query.embedding_utils import get_top_k_embeddings
from gpt_index.indices.query.list.query import BaseGPTListIndexQuery


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
            **kwargs,
        )
        self._embed_model = embed_model or OpenAIEmbedding()
        self.similarity_top_k = similarity_top_k

    def _get_nodes_for_response(
        self, query_str: str, verbose: bool = False
    ) -> List[Node]:
        """Get nodes for response."""
        nodes = self.index_struct.nodes
        # top k nodes
        query_embedding, node_embeddings = self._get_embeddings(query_str, nodes)

        _, top_idxs = get_top_k_embeddings(
            self._embed_model,
            query_embedding,
            node_embeddings,
            similarity_top_k=self.similarity_top_k,
            embedding_ids=list(range(len(nodes))),
        )

        top_k_nodes = [nodes[i] for i in top_idxs]
        if verbose:
            top_k_node_text = "\n".join([n.get_text() for n in top_k_nodes])
            print(f"Top {len(top_idxs)} nodes:\n{top_k_node_text}")
        return top_k_nodes

    def _get_embeddings(
        self, query_str: str, nodes: List[Node]
    ) -> Tuple[List[float], List[List[float]]]:
        """Get top nodes by similarity to the query."""
        query_embedding = self._embed_model.get_query_embedding(query_str)
        node_embeddings: List[List[float]] = []
        for node in self.index_struct.nodes:
            if node.embedding is not None:
                text_embedding = node.embedding
            else:
                text_embedding = self._embed_model.get_text_embedding(node.get_text())
                node.embedding = text_embedding

            node_embeddings.append(text_embedding)
        return query_embedding, node_embeddings
