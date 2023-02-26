"""GPT Tree Index inserter."""

from typing import Optional

from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.utils import extract_numbers_given_response, get_sorted_node_list
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TextSplitter
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_INSERT_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)
from gpt_index.schema import BaseDocument


class GPTIndexInserter:
    """LlamaIndex inserter."""

    def __init__(
        self,
        index_graph: IndexGraph,
        llm_predictor: LLMPredictor,
        prompt_helper: PromptHelper,
        text_splitter: TextSplitter,
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
        self._llm_predictor = llm_predictor
        self._prompt_helper = prompt_helper
        self._text_splitter = text_splitter

    def _insert_under_parent_and_consolidate(
        self, text_chunk: str, doc: BaseDocument, parent_node: Optional[Node]
    ) -> None:
        """Insert node under parent and consolidate.

        Consolidation will happen by dividing up child nodes, and creating a new
        intermediate layer of nodes.

        """
        # perform insertion
        text_node = Node(
            text=text_chunk,
            index=self.index_graph.size,
            ref_doc_id=doc.get_doc_id(),
            embedding=doc.embedding,
            extra_info=doc.extra_info,
        )
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

            text_chunk1 = self._prompt_helper.get_text_from_nodes(
                half1, prompt=self.summary_prompt
            )
            summary1, _ = self._llm_predictor.predict(
                self.summary_prompt, context_str=text_chunk1
            )
            node1 = Node(
                text=summary1,
                index=cur_node_index,
                child_indices={n.index for n in half1},
            )

            text_chunk2 = self._prompt_helper.get_text_from_nodes(
                half2, prompt=self.summary_prompt
            )
            summary2, _ = self._llm_predictor.predict(
                self.summary_prompt, context_str=text_chunk2
            )
            node2 = Node(
                text=summary2,
                index=cur_node_index + 1,
                child_indices={n.index for n in half2},
            )

            # insert half1 and half2 as new children of parent_node
            # first remove child indices from parent node
            if parent_node is not None:
                parent_node.child_indices = set()
            else:
                self.index_graph.root_nodes = {}
            self.index_graph.insert_under_parent(node1, parent_node)
            self.index_graph.insert_under_parent(node2, parent_node)

    def _insert_node(
        self, text_chunk: str, doc: BaseDocument, parent_node: Optional[Node]
    ) -> None:
        """Insert node."""
        cur_graph_nodes = self.index_graph.get_children(parent_node)
        cur_graph_node_list = get_sorted_node_list(cur_graph_nodes)
        # if cur_graph_nodes is empty (start with empty graph), then insert under
        # parent (insert new root node)
        if len(cur_graph_nodes) == 0:
            self._insert_under_parent_and_consolidate(text_chunk, doc, parent_node)
        # check if leaf nodes, then just insert under parent
        elif len(cur_graph_node_list[0].child_indices) == 0:
            self._insert_under_parent_and_consolidate(text_chunk, doc, parent_node)
        # else try to find the right summary node to insert under
        else:
            numbered_text = self._prompt_helper.get_numbered_text_from_nodes(
                cur_graph_node_list, prompt=self.insert_prompt
            )
            response, _ = self._llm_predictor.predict(
                self.insert_prompt,
                new_chunk_text=text_chunk,
                num_chunks=len(cur_graph_node_list),
                context_list=numbered_text,
            )
            numbers = extract_numbers_given_response(response)
            if numbers is None or len(numbers) == 0:
                # NOTE: if we can't extract a number, then we just insert under parent
                self._insert_under_parent_and_consolidate(text_chunk, doc, parent_node)
            elif int(numbers[0]) > len(cur_graph_node_list):
                # NOTE: if number is out of range, then we just insert under parent
                self._insert_under_parent_and_consolidate(text_chunk, doc, parent_node)
            else:
                selected_node = cur_graph_node_list[int(numbers[0]) - 1]
                self._insert_node(text_chunk, doc, selected_node)

        # now we need to update summary for parent node, since we
        # need to bubble updated summaries up the tree
        if parent_node is not None:
            # refetch children
            cur_graph_nodes = self.index_graph.get_children(parent_node)
            cur_graph_node_list = get_sorted_node_list(cur_graph_nodes)
            text_chunk = self._prompt_helper.get_text_from_nodes(
                cur_graph_node_list, prompt=self.summary_prompt
            )
            new_summary, _ = self._llm_predictor.predict(
                self.summary_prompt, context_str=text_chunk
            )

            parent_node.text = new_summary

    def insert(self, doc: BaseDocument) -> None:
        """Insert into index_graph."""
        text_chunks = self._text_splitter.split_text(doc.get_text())

        for text_chunk in text_chunks:
            self._insert_node(text_chunk, doc, None)
