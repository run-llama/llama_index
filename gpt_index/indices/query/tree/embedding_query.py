"""Query Tree using embedding similarity between query and node text."""

from typing import Any, Dict, List, Optional, Tuple

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.query.tree.leaf_query import GPTTreeIndexLeafQuery
from gpt_index.indices.utils import get_sorted_node_list
from gpt_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    TreeSelectMultiplePrompt,
    TreeSelectPrompt,
)


class GPTTreeIndexEmbeddingQuery(GPTTreeIndexLeafQuery):
    """
    GPT Tree Index embedding query.

    This class traverses the index graph using the embedding similarity between the
    query and the node text.

    .. code-block:: python

        response = index.query("<query_str>", mode="embedding")

    Args:
        query_template (Optional[TreeSelectPrompt]): Tree Select Query Prompt
            (see :ref:`Prompt-Templates`).
        query_template_multiple (Optional[TreeSelectMultiplePrompt]): Tree Select
            Query Prompt (Multiple)
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[QuestionAnswerPrompt]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): Refinement Prompt
            (see :ref:`Prompt-Templates`).
        child_branch_factor (int): Number of child nodes to consider at each level.
            If child_branch_factor is 1, then the query will only choose one child node
            to traverse for any given parent node.
            If child_branch_factor is 2, then the query will choose two child nodes.
        embed_model (Optional[OpenAIEmbedding]): Embedding model to use for
            embedding similarity.

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        query_template: Optional[TreeSelectPrompt] = None,
        query_template_multiple: Optional[TreeSelectMultiplePrompt] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        child_branch_factor: int = 1,
        embed_model: Optional[OpenAIEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct,
            query_template=query_template,
            query_template_multiple=query_template_multiple,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            child_branch_factor=child_branch_factor,
            **kwargs,
        )
        self._embed_model = embed_model or OpenAIEmbedding()
        self.child_branch_factor = child_branch_factor

    def _query_level(
        self,
        cur_nodes: Dict[int, Node],
        query_str: str,
        level: int = 0,
        verbose: bool = False,
    ) -> str:

        cur_node_list = get_sorted_node_list(cur_nodes)

        # Get the node with the highest similarity to the query
        selected_node, selected_index = self._get_most_similar_node(
            cur_node_list, query_str
        )
        if verbose:
            print(
                f">[Level {level}] Node [{selected_index+1}] Summary text: "
                f"{' '.join(selected_node.get_text().splitlines())}"
            )

        # Get the response for the selected node
        response = self._query_with_selected_node(
            selected_node, query_str, level=level, verbose=verbose
        )

        return response

    def _get_query_text_embedding_similarities(
        self, query_str: str, nodes: List[Node]
    ) -> List[float]:
        """
        Get query text embedding similarity.

        Cache the query embedding and the node text embedding.

        """
        query_embedding = self._embed_model.get_query_embedding(query_str)
        similarities = []
        for node in nodes:
            if node.embedding is not None:
                text_embedding = node.embedding
            else:
                text_embedding = self._embed_model.get_text_embedding(node.get_text())
                node.embedding = text_embedding

            similarity = self._embed_model.similarity(query_embedding, text_embedding)
            similarities.append(similarity)
        return similarities

    def _get_most_similar_node(
        self, nodes: List[Node], query_str: str
    ) -> Tuple[Node, int]:
        """Get the node with the highest similarity to the query."""
        similarities = self._get_query_text_embedding_similarities(query_str, nodes)

        selected_index = similarities.index(max(similarities))

        selected_node = nodes[similarities.index(max(similarities))]
        return selected_node, selected_index
