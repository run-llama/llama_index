"""Common classes/functions for tree index operations."""


import asyncio
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from gpt_index.async_utils import run_async_tasks
from gpt_index.data_structs.data_structs import Node
from gpt_index.data_structs.data_structs_v2 import IndexGraph
from gpt_index.docstore import DocumentStore
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.utils import get_sorted_node_list, truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.logger.base import LlamaLogger
from gpt_index.prompts.prompts import SummaryPrompt

logger = logging.getLogger(__name__)


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
        docstore: Optional[DocumentStore] = None,
        use_async: bool = False,
        llama_logger: Optional[LlamaLogger] = None,
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        self._llm_predictor = llm_predictor
        self._prompt_helper = prompt_helper
        self._use_async = use_async
        self._docstore = docstore or DocumentStore()
        # print("prepre init docstore: ", docstore)
        # if docstore is None:
        #     print("wtf")
        # self._docstore = docstore or DocumentStore()
        # if docstore != self._docstore:
        #     print("wtf2")

        print("pre init docstore: ", self._docstore)
        self._llama_logger = llama_logger or LlamaLogger()

    @property
    def docstore(self) -> DocumentStore:
        """Return docstore."""
        return self._docstore

    def build_from_nodes(
        self,
        nodes: Sequence[Node],
        build_tree: bool = True,
    ) -> IndexGraph:
        """Build from text.

        Returns:
            IndexGraph: graph object consisting of all_nodes, root_nodes

        """
        all_nodes: Dict[int, str] = {}
        all_nodes.update({node.index: node.get_doc_id() for node in nodes})

        if build_tree:
            # instantiate all_nodes from initial text chunks
            root_nodes = self.build_index_from_nodes(all_nodes, all_nodes, level=0)
        else:
            # if build_tree is False, then don't surface any root nodes
            root_nodes = {}
        print("current docstore0", self._docstore)
        return IndexGraph(all_nodes=all_nodes, root_nodes=root_nodes)

    def _prepare_node_and_text_chunks(
        self, cur_node_ids: Dict[int, str]
    ) -> Tuple[List[int], List[List[Node]], List[str]]:
        """Prepare node and text chunks."""
        cur_nodes = {
            index: self._docstore.get_node(node_id)
            for index, node_id in cur_node_ids.items()
        }
        cur_node_list = get_sorted_node_list(cur_nodes)
        logger.info(
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
        return indices, cur_nodes_chunks, text_chunks

    def _construct_parent_nodes(
        self,
        cur_index: int,
        indices: List[int],
        cur_nodes_chunks: List[List[Node]],
        summaries: List[str],
    ) -> Dict[int, str]:
        """Construct parent nodes.

        Save nodes to docstore.

        """
        new_node_dict = {}
        for i, cur_nodes_chunk, new_summary in zip(
            indices, cur_nodes_chunks, summaries
        ):
            logger.debug(
                f"> {i}/{len(cur_nodes_chunk)}, "
                f"summary: {truncate_text(new_summary, 50)}"
            )
            new_node = Node(
                text=new_summary,
                index=cur_index,
                child_indices={n.index for n in cur_nodes_chunk},
            )
            new_node_dict[cur_index] = new_node.get_doc_id()
            self._docstore.add_documents([new_node])
            cur_index += 1
        return new_node_dict

    def build_index_from_nodes(
        self, cur_node_ids: Dict[int, str], all_node_ids: Dict[int, str], level: int = 0
    ) -> Dict[int, str]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_index = len(all_node_ids)
        indices, cur_nodes_chunks, text_chunks = self._prepare_node_and_text_chunks(
            cur_node_ids
        )

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
        self._llama_logger.add_log({"summaries": summaries, "level": level})

        new_node_dict = self._construct_parent_nodes(
            cur_index, indices, cur_nodes_chunks, summaries
        )
        all_node_ids.update(new_node_dict)

        if len(new_node_dict) <= self.num_children:
            return new_node_dict
        else:
            return self.build_index_from_nodes(
                new_node_dict, all_node_ids, level=level + 1
            )

    async def abuild_index_from_nodes(
        self, cur_node_ids: Dict[int, str], all_node_ids: Dict[int, str], level: int = 0
    ) -> Dict[int, str]:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        cur_index = len(all_node_ids)
        indices, cur_nodes_chunks, text_chunks = self._prepare_node_and_text_chunks(
            cur_node_ids
        )

        tasks = [
            self._llm_predictor.apredict(self.summary_prompt, context_str=text_chunk)
            for text_chunk in text_chunks
        ]
        outputs: List[Tuple[str, str]] = await asyncio.gather(*tasks)
        summaries = [output[0] for output in outputs]
        self._llama_logger.add_log({"summaries": summaries, "level": level})

        new_node_dict = self._construct_parent_nodes(
            cur_index, indices, cur_nodes_chunks, summaries
        )
        all_node_ids.update(new_node_dict)

        if len(new_node_dict) <= self.num_children:
            return new_node_dict
        else:
            return await self.abuild_index_from_nodes(
                new_node_dict, all_node_ids, level=level + 1
            )
