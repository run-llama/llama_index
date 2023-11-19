"""Query Tree using embedding similarity between query and node text."""

import logging
from typing import Dict, List, Tuple, cast

from llama_index.indices.tree.select_leaf_retriever import TreeSelectLeafRetriever
from llama_index.indices.utils import get_sorted_node_list
from llama_index.schema import BaseNode, MetadataMode, QueryBundle

logger = logging.getLogger(__name__)


class TreeSelectLeafEmbeddingRetriever(TreeSelectLeafRetriever):
    """Tree select leaf embedding retriever.

    This class traverses the index graph using the embedding similarity between the
    query and the node text.

    Args:
        query_template (Optional[BasePromptTemplate]): Tree Select Query Prompt
            (see :ref:`Prompt-Templates`).
        query_template_multiple (Optional[BasePromptTemplate]): Tree Select
            Query Prompt (Multiple)
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[BasePromptTemplate]): Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        refine_template (Optional[BasePromptTemplate]): Refinement Prompt
            (see :ref:`Prompt-Templates`).
        child_branch_factor (int): Number of child nodes to consider at each level.
            If child_branch_factor is 1, then the query will only choose one child node
            to traverse for any given parent node.
            If child_branch_factor is 2, then the query will choose two child nodes.
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.

    """

    def _query_level(
        self,
        cur_node_ids: Dict[int, str],
        query_bundle: QueryBundle,
        level: int = 0,
    ) -> str:
        """Answer a query recursively."""
        cur_nodes = {
            index: self._docstore.get_node(node_id)
            for index, node_id in cur_node_ids.items()
        }
        cur_node_list = get_sorted_node_list(cur_nodes)

        # Get the node with the highest similarity to the query
        selected_nodes, selected_indices = self._get_most_similar_nodes(
            cur_node_list, query_bundle
        )

        result_response = None
        for node, index in zip(selected_nodes, selected_indices):
            logger.debug(
                f">[Level {level}] Node [{index+1}] Summary text: "
                f"{' '.join(node.get_content().splitlines())}"
            )

            # Get the response for the selected node
            result_response = self._query_with_selected_node(
                node, query_bundle, level=level, prev_response=result_response
            )

        return cast(str, result_response)

    def _get_query_text_embedding_similarities(
        self, query_bundle: QueryBundle, nodes: List[BaseNode]
    ) -> List[float]:
        """
        Get query text embedding similarity.

        Cache the query embedding and the node text embedding.

        """
        if query_bundle.embedding is None:
            query_bundle.embedding = (
                self._service_context.embed_model.get_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )
        similarities = []
        for node in nodes:
            if node.embedding is None:
                node.embedding = self._service_context.embed_model.get_text_embedding(
                    node.get_content(metadata_mode=MetadataMode.EMBED)
                )

            similarity = self._service_context.embed_model.similarity(
                query_bundle.embedding, node.embedding
            )
            similarities.append(similarity)
        return similarities

    def _get_most_similar_nodes(
        self, nodes: List[BaseNode], query_bundle: QueryBundle
    ) -> Tuple[List[BaseNode], List[int]]:
        """Get the node with the highest similarity to the query."""
        similarities = self._get_query_text_embedding_similarities(query_bundle, nodes)

        selected_nodes: List[BaseNode] = []
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

    def _select_nodes(
        self,
        cur_node_list: List[BaseNode],
        query_bundle: QueryBundle,
        level: int = 0,
    ) -> List[BaseNode]:
        selected_nodes, _ = self._get_most_similar_nodes(cur_node_list, query_bundle)
        return selected_nodes
