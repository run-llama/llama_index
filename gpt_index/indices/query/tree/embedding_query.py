"""Query Tree using embedding similarity between query and node text."""

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.query.tree.leaf_query import GPTTreeIndexLeafQuery
from gpt_index.indices.utils import get_sorted_node_list
from gpt_index.prompts.prompts import TreeSelectMultiplePrompt, TreeSelectPrompt


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
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        query_template: Optional[TreeSelectPrompt] = None,
        query_template_multiple: Optional[TreeSelectMultiplePrompt] = None,
        child_branch_factor: int = 1,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct,
            query_template=query_template,
            query_template_multiple=query_template_multiple,
            child_branch_factor=child_branch_factor,
            embed_model=embed_model,
            **kwargs,
        )
        self.child_branch_factor = child_branch_factor

    def _query_level(
        self,
        cur_nodes: Dict[int, Node],
        query_bundle: QueryBundle,
        level: int = 0,
    ) -> str:
        cur_node_list = get_sorted_node_list(cur_nodes)

        # Get the node with the highest similarity to the query
        selected_nodes, selected_indices = self._get_most_similar_nodes(
            cur_node_list, query_bundle
        )

        result_response = None
        for node, index in zip(selected_nodes, selected_indices):
            logging.debug(
                f">[Level {level}] Node [{index+1}] Summary text: "
                f"{' '.join(node.get_text().splitlines())}"
            )

            # Get the response for the selected node
            result_response = self._query_with_selected_node(
                node, query_bundle, level=level, prev_response=result_response
            )

        return cast(str, result_response)

    def _get_query_text_embedding_similarities(
        self, query_bundle: QueryBundle, nodes: List[Node]
    ) -> List[float]:
        """
        Get query text embedding similarity.

        Cache the query embedding and the node text embedding.

        """
        query_embedding = self._embed_model.get_agg_embedding_from_queries(
            query_bundle.embedding_strs
        )
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

    def _get_most_similar_nodes(
        self, nodes: List[Node], query_bundle: QueryBundle
    ) -> Tuple[List[Node], List[int]]:
        """Get the node with the highest similarity to the query."""
        similarities = self._get_query_text_embedding_similarities(query_bundle, nodes)

        selected_nodes: List[Node] = []
        selected_indices: List[int] = []
        for node, _ in sorted(
            zip(nodes, similarities), key=lambda x: x[1], reverse=True
        ):
            if len(selected_nodes) < self.child_branch_factor:
                selected_nodes.append(node)
                selected_indices.append(nodes.index(node))
            else:
                break

        return selected_nodes, selected_indices
