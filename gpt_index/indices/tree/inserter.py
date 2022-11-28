"""GPT Tree Index inserter."""

from typing import Optional

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.utils import (
    extract_numbers_given_response,
    get_chunk_size_given_prompt,
    get_numbered_text_from_nodes,
    get_sorted_node_list,
    get_text_from_nodes,
)
from gpt_index.langchain_helpers.chain_wrapper import openai_llm_predict
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_INSERT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)


class GPTIndexInserter:
    """GPT Index inserter."""

    def __init__(
        self,
        index_graph: IndexGraph,
        num_children: int = 10,
        insert_prompt: Prompt = DEFAULT_INSERT_PROMPT,
        summary_prompt: Prompt = DEFAULT_SUMMARY_PROMPT,
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        self.insert_prompt = insert_prompt
        self.index_graph = index_graph
        chunk_size = get_chunk_size_given_prompt(
            summary_prompt.format(text=""), MAX_CHUNK_SIZE, num_children, NUM_OUTPUTS
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=MAX_CHUNK_OVERLAP // num_children,
        )

    def _insert_under_parent_and_consolidate(
        self, text_chunk: str, parent_node: Optional[Node]
    ) -> None:
        """Insert node under parent and consolidate.

        Consolidation will happen by dividing up child nodes, and creating a new
        intermediate layer of nodes.

        """
        # perform insertion
        text_node = Node(text_chunk, self.index_graph.size, set())
        self.index_graph.insert_under_parent(text_node, parent_node)

        # if under num_children limit, then we're fine
        if len(self.index_graph.get_children(parent_node)) <= self.num_children:
            return
        else:
            # perform consolidation
            cur_graph_nodes = self.index_graph.get_children(parent_node)
            cur_graph_node_list = get_sorted_node_list(cur_graph_nodes)
            # this layer is all leaf nodes, consolidate and split leaf nodes
            cur_node_index = self.index_graph.size
            # consolidate and split leaf nodes in half
            # TODO: do better splitting (with a GPT prompt etc.)
            half1 = cur_graph_node_list[: len(cur_graph_nodes) // 2]
            half2 = cur_graph_node_list[len(cur_graph_nodes) // 2 :]

            text_chunk1 = get_text_from_nodes(half1)
            summary1, _ = openai_llm_predict(self.summary_prompt, text=text_chunk1)
            node1 = Node(summary1, cur_node_index, {n.index for n in half1})

            text_chunk2 = get_text_from_nodes(half2)
            summary2, _ = openai_llm_predict(self.summary_prompt, text=text_chunk2)
            node2 = Node(summary2, cur_node_index + 1, {n.index for n in half2})

            # insert half1 and half2 as new children of parent_node
            # first remove child indices from parent node
            if parent_node is not None:
                parent_node.child_indices = set()
            else:
                self.index_graph.root_nodes = {}
            self.index_graph.insert_under_parent(node1, parent_node)
            self.index_graph.insert_under_parent(node2, parent_node)

    def _insert_node(self, text_chunk: str, parent_node: Optional[Node]) -> None:
        """Insert node."""
        cur_graph_nodes = self.index_graph.get_children(parent_node)
        cur_graph_node_list = get_sorted_node_list(cur_graph_nodes)
        # if cur_graph_nodes is empty (start with empty graph), then insert under
        # parent (insert new root node)
        if len(cur_graph_nodes) == 0:
            self._insert_under_parent_and_consolidate(text_chunk, parent_node)
        # check if leaf nodes, then just insert under parent
        elif len(cur_graph_node_list[0].child_indices) == 0:
            self._insert_under_parent_and_consolidate(text_chunk, parent_node)
        # else try to find the right summary node to insert under
        else:
            response, _ = openai_llm_predict(
                self.insert_prompt,
                new_chunk_text=text_chunk,
                num_chunks=len(cur_graph_node_list),
                context_list=get_numbered_text_from_nodes(cur_graph_node_list),
            )
            numbers = extract_numbers_given_response(response)
            if numbers is None or len(numbers) == 0:
                # NOTE: if we can't extract a number, then we just insert under parent
                self._insert_under_parent_and_consolidate(text_chunk, parent_node)
            elif int(numbers[0]) > len(cur_graph_node_list):
                # NOTE: if number is out of range, then we just insert under parent
                self._insert_under_parent_and_consolidate(text_chunk, parent_node)
            else:
                selected_node = cur_graph_node_list[int(numbers[0]) - 1]
                self._insert_node(text_chunk, selected_node)

        # now we need to update summary for parent node, since we
        # need to bubble updated summaries up the tree
        if parent_node is not None:
            # refetch children
            cur_graph_nodes = self.index_graph.get_children(parent_node)
            cur_graph_node_list = get_sorted_node_list(cur_graph_nodes)
            text_chunk = get_text_from_nodes(cur_graph_node_list)
            new_summary, _ = openai_llm_predict(self.summary_prompt, text=text_chunk)

            parent_node.text = new_summary

    def insert(self, text: str) -> None:
        """Insert into index_graph."""
        text_chunks = self.text_splitter.split_text(text)

        for text_chunk in text_chunks:
            self._insert_node(text_chunk, None)
