"""Embedding query for list index."""
from typing import Any, List, Optional

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.data_structs import IndexList, Node
from gpt_index.indices.query.list.query import BaseGPTListIndexQuery
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt


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
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        similarity_top_k: Optional[int] = 1,
        embed_model: Optional[OpenAIEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
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
        similarities = self._get_query_text_embedding_similarities(query_str, nodes)
        sorted_node_tups = sorted(
            zip(similarities, nodes), key=lambda x: x[0], reverse=True
        )
        sorted_nodes = [n for _, n in sorted_node_tups]
        similarity_top_k = self.similarity_top_k or len(nodes)
        top_k_nodes = sorted_nodes[:similarity_top_k]
        if verbose:
            top_k_node_text = "\n".join([n.get_text() for n in top_k_nodes])
            print(f"Top {similarity_top_k} nodes:\n{top_k_node_text}")
        return top_k_nodes

    def _get_query_text_embedding_similarities(
        self, query_str: str, nodes: List[Node]
    ) -> List[float]:
        """Get top nodes by similarity to the query."""
        query_embedding = self._embed_model.get_query_embedding(query_str)
        similarities = []
        for node in self.index_struct.nodes:
            if node.embedding is not None:
                text_embedding = node.embedding
            else:
                text_embedding = self._embed_model.get_text_embedding(node.get_text())
                node.embedding = text_embedding

            similarity = self._embed_model.similarity(query_embedding, text_embedding)
            similarities.append(similarity)

        return similarities
