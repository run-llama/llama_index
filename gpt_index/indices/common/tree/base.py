"""Common classes/functions for tree index operations."""


import logging
from typing import Dict, List, Sequence, Tuple

from gpt_index.async_utils import run_async_tasks
from gpt_index.data_structs.data_structs import IndexGraph, Node
from gpt_index.indices.node_utils import get_text_splits_from_document
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.utils import get_sorted_node_list, truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TextSplitter
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
        text_splitter: TextSplitter,
        use_async: bool = False,
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        self._llm_predictor = llm_predictor
        self._prompt_helper = prompt_helper
        self._text_splitter = text_splitter
        self._use_async = use_async

    def _get_nodes_from_document(
        self, start_idx: int, document: BaseDocument
    ) -> Dict[int, Node]:
        """Add document to index."""
        # NOTE: summary prompt does not need to be partially formatted
        text_splits = get_text_splits_from_document(
            document=document, text_splitter=self._text_splitter
        )
        text_chunks = [text_split.text_chunk for text_split in text_splits]
        doc_nodes = {
            (start_idx + i): Node(
                text=t,
                index=(start_idx + i),
                ref_doc_id=document.get_doc_id(),
                embedding=document.embedding,
                extra_info=document.extra_info,
            )
            for i, t in enumerate(text_chunks)
        }
        return doc_nodes

    def build_from_text(
        self,
        documents: Sequence[BaseDocument],
        build_tree: bool = True,
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
            root_nodes = self.build_index_from_nodes(all_nodes, all_nodes)
        else:
            # if build_tree is False, then don't surface any root nodes
            root_nodes = {}
        return IndexGraph(all_nodes=all_nodes, root_nodes=root_nodes)

    def build_index_from_nodes(
        self,
        cur_nodes: Dict[int, Node],
        all_nodes: Dict[int, Node],
    ) -> Dict[int, Node]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_node_list = get_sorted_node_list(cur_nodes)
        cur_index = len(all_nodes)
        new_node_dict = {}
        logging.info(
            f"> Building index from nodes: {len(cur_nodes) // self.num_children} chunks"
        )
        indices, cur_nodes_chunks, text_chunks = [], [], []
        for i in range(0, len(cur_node_list), self.num_children):
            cur_nodes_chunk = cur_node_list[i : i + self.num_children]
            text_chunk = self._prompt_helper.get_text_from_nodes(
                cur_nodes_chunk, prompt=self.summary_prompt
            )
            indices.append(i)
            cur_nodes_chunks.append(cur_nodes_chunk)
            text_chunks.append(text_chunk)

        if self._use_async:
            tasks = [
                self._llm_predictor.apredict(
                    self.summary_prompt, context_str=text_chunk
                )
                for text_chunk in text_chunks
            ]
            outputs: List[Tuple[str, str]] = run_async_tasks(tasks)
            summaries = [output[0] for output in outputs]
        else:
            summaries = [
                self._llm_predictor.predict(
                    self.summary_prompt, context_str=text_chunk
                )[0]
                for text_chunk in text_chunks
            ]

        for i, cur_nodes_chunk, new_summary in zip(
            indices, cur_nodes_chunks, summaries
        ):
            logging.debug(
                f"> {i}/{len(cur_nodes)}, summary: {truncate_text(new_summary, 50)}"
            )
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
