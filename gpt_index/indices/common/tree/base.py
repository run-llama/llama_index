"""Common classes/functions for tree index operations."""


from typing import Dict, Sequence

from gpt_index.indices.data_structs import IndexGraph, Node
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.utils import get_sorted_node_list, truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.prompts import SummaryPrompt
from gpt_index.schema import BaseDocument


class GPTTreeIndexBuilder:
    """GPT tree index builder.

    Helper class to build the tree-structured index,
    or to synthesize an answer.

    """

    def __init__(
        self,
        num_children: int,
        summary_prompt: SummaryPrompt,
        llm_predictor: LLMPredictor,
        prompt_helper: PromptHelper,
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        self._llm_predictor = llm_predictor
        self._prompt_helper = prompt_helper
        self._text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.summary_prompt, self.num_children
        )

    def _get_nodes_from_document(
        self, start_idx: int, document: BaseDocument
    ) -> Dict[int, Node]:
        """Add document to index."""
        text_chunks = self._text_splitter.split_text(document.get_text())
        doc_nodes = {
            (start_idx + i): Node(
                text=t, index=(start_idx + i), ref_doc_id=document.get_doc_id()
            )
            for i, t in enumerate(text_chunks)
        }
        return doc_nodes

    def build_from_text(
        self,
        documents: Sequence[BaseDocument],
        build_tree: bool = True,
        verbose: bool = False,
    ) -> IndexGraph:
        """Build from text.

        Returns:
            IndexGraph: graph object consisting of all_nodes, root_nodes

        """
        all_nodes: Dict[int, Node] = {}
        for d in documents:
            all_nodes.update(self._get_nodes_from_document(len(all_nodes), d))

        if build_tree:
            # instantiate all_nodes from initial text chunks
            root_nodes = self.build_index_from_nodes(
                all_nodes, all_nodes, verbose=verbose
            )
        else:
            # if build_tree is False, then don't surface any root nodes
            root_nodes = {}
        return IndexGraph(all_nodes=all_nodes, root_nodes=root_nodes)

    def build_index_from_nodes(
        self,
        cur_nodes: Dict[int, Node],
        all_nodes: Dict[int, Node],
        verbose: bool = False,
    ) -> Dict[int, Node]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_node_list = get_sorted_node_list(cur_nodes)
        cur_index = len(all_nodes)
        new_node_dict = {}
        print(
            f"> Building index from nodes: {len(cur_nodes) // self.num_children} chunks"
        )
        for i in range(0, len(cur_node_list), self.num_children):
            print(f"{i}/{len(cur_nodes)}")
            cur_nodes_chunk = cur_node_list[i : i + self.num_children]
            text_chunk = self._prompt_helper.get_text_from_nodes(
                cur_nodes_chunk, prompt=self.summary_prompt
            )

            new_summary, _ = self._llm_predictor.predict(
                self.summary_prompt, context_str=text_chunk
            )

            if verbose:
                fmt_summary = truncate_text(new_summary, 50)
                print(f"> {i}/{len(cur_nodes)}, summary: {fmt_summary}")
            new_node = Node(
                text=new_summary,
                index=cur_index,
                child_indices={n.index for n in cur_nodes_chunk},
            )
            new_node_dict[cur_index] = new_node
            cur_index += 1

        all_nodes.update(new_node_dict)

        if len(new_node_dict) <= self.num_children:
            return new_node_dict
        else:
            return self.build_index_from_nodes(new_node_dict, all_nodes)
