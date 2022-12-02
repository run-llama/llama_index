"""Query Tree using embedding similarity between query and node text."""

from typing import Dict, List

from gpt_index.embeddings.utils import (
    TEXT_SEARCH_MODE,
    get_query_text_embedding_similarity,
)
from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.tree.leaf_query import GPTTreeIndexLeafQuery
from gpt_index.indices.utils import get_sorted_node_list
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_QUERY_PROMPT,
    DEFAULT_QUERY_PROMPT_MULTIPLE,
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)

EMBEDDING_MODE = "embedding"


class GPTTreeIndexEmbeddingQuery(GPTTreeIndexLeafQuery):
    """
    GPT Tree Index embedding query.

    This class traverses the index graph using the embedding similarity between the
    query and the node text.

    """

    def __init__(
        self,
        index_struct: IndexGraph,
        query_template: Prompt = DEFAULT_QUERY_PROMPT,
        query_template_multiple: Prompt = DEFAULT_QUERY_PROMPT_MULTIPLE,
        text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
        refine_template: Prompt = DEFAULT_REFINE_PROMPT,
        child_branch_factor: int = 1,
    ) -> None:
        """Initialize params."""
        super().__init__(
            index_struct,
            query_template,
            query_template_multiple,
            text_qa_template,
            refine_template,
            child_branch_factor,
        )
        self.child_branch_factor = child_branch_factor

    def query(
        self, query_str: str, verbose: bool = False, mode: str = TEXT_SEARCH_MODE
    ) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        return self._query(
            self.index_struct.root_nodes, query_str, level=0, verbose=verbose, mode=mode
        ).strip()

    def _query(
        self,
        cur_nodes: Dict[int, Node],
        query_str: str,
        level: int = 0,
        verbose: bool = False,
        mode: str = TEXT_SEARCH_MODE,
    ) -> str:

        cur_node_list = get_sorted_node_list(cur_nodes)

        # Get the node with the highest similarity to the query
        selected_node = self._get_most_similar_node(cur_node_list, query_str, mode)

        # Get the response for the selected node
        response = self._query_with_selected_node(
            selected_node, query_str, level=level, verbose=verbose
        )

        return response

    def _get_most_similar_node(
        self, nodes: List[Node], query_str: str, mode: str = TEXT_SEARCH_MODE
    ) -> Node:
        """Get the node with the highest similarity to the query."""
        similarities = [
            get_query_text_embedding_similarity(query_str, node.text, mode)
            for node in nodes
        ]
        selected_node = nodes[similarities.index(max(similarities))]
        return selected_node
