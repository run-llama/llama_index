"""Common classes/functions for tree index operations."""


import asyncio
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from llama_index.async_utils import run_async_tasks
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.data_structs.data_structs import IndexGraph
from llama_index.indices.utils import get_sorted_node_list, truncate_text
from llama_index.prompts import BasePromptTemplate
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.service_context import ServiceContext
from llama_index.storage.docstore import BaseDocumentStore
from llama_index.storage.docstore.registry import get_default_docstore
from llama_index.utils import get_tqdm_iterable

logger = logging.getLogger(__name__)


class GPTTreeIndexBuilder:
    """GPT tree index builder.

    Helper class to build the tree-structured index,
    or to synthesize an answer.

    """

    def __init__(
        self,
        num_children: int,
        summary_prompt: BasePromptTemplate,
        service_context: ServiceContext,
        docstore: Optional[BaseDocumentStore] = None,
        show_progress: bool = False,
        use_async: bool = False,
    ) -> None:
        """Initialize with params."""
        if num_children < 2:
            raise ValueError("Invalid number of children.")
        self.num_children = num_children
        self.summary_prompt = summary_prompt
        self._service_context = service_context
        self._use_async = use_async
        self._show_progress = show_progress
        self._docstore = docstore or get_default_docstore()

    @property
    def docstore(self) -> BaseDocumentStore:
        """Return docstore."""
        return self._docstore

    def build_from_nodes(
        self,
        nodes: Sequence[BaseNode],
        build_tree: bool = True,
    ) -> IndexGraph:
        """Build from text.

        Returns:
            IndexGraph: graph object consisting of all_nodes, root_nodes

        """
        index_graph = IndexGraph()
        for node in nodes:
            index_graph.insert(node)

        if build_tree:
            return self.build_index_from_nodes(
                index_graph, index_graph.all_nodes, index_graph.all_nodes, level=0
            )
        else:
            return index_graph

    def _prepare_node_and_text_chunks(
        self, cur_node_ids: Dict[int, str]
    ) -> Tuple[List[int], List[List[BaseNode]], List[str]]:
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
            truncated_chunks = self._service_context.prompt_helper.truncate(
                prompt=self.summary_prompt,
                text_chunks=[
                    node.get_content(metadata_mode=MetadataMode.LLM)
                    for node in cur_nodes_chunk
                ],
            )
            text_chunk = "\n".join(truncated_chunks)
            indices.append(i)
            cur_nodes_chunks.append(cur_nodes_chunk)
            text_chunks.append(text_chunk)
        return indices, cur_nodes_chunks, text_chunks

    def _construct_parent_nodes(
        self,
        index_graph: IndexGraph,
        indices: List[int],
        cur_nodes_chunks: List[List[BaseNode]],
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
            new_node = TextNode(text=new_summary)
            index_graph.insert(new_node, children_nodes=cur_nodes_chunk)
            index = index_graph.get_index(new_node)
            new_node_dict[index] = new_node.node_id
            self._docstore.add_documents([new_node], allow_update=False)
        return new_node_dict

    def build_index_from_nodes(
        self,
        index_graph: IndexGraph,
        cur_node_ids: Dict[int, str],
        all_node_ids: Dict[int, str],
        level: int = 0,
    ) -> IndexGraph:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        if len(cur_node_ids) <= self.num_children:
            index_graph.root_nodes = cur_node_ids
            return index_graph

        indices, cur_nodes_chunks, text_chunks = self._prepare_node_and_text_chunks(
            cur_node_ids
        )

        with self._service_context.callback_manager.event(
            CBEventType.TREE, payload={EventPayload.CHUNKS: text_chunks}
        ) as event:
            if self._use_async:
                tasks = [
                    self._service_context.llm.apredict(
                        self.summary_prompt, context_str=text_chunk
                    )
                    for text_chunk in text_chunks
                ]
                outputs: List[Tuple[str, str]] = run_async_tasks(
                    tasks,
                    show_progress=self._show_progress,
                    progress_bar_desc="Generating summaries",
                )
                summaries = [output[0] for output in outputs]
            else:
                text_chunks_progress = get_tqdm_iterable(
                    text_chunks,
                    show_progress=self._show_progress,
                    desc="Generating summaries",
                )
                summaries = [
                    self._service_context.llm.predict(
                        self.summary_prompt, context_str=text_chunk
                    )
                    for text_chunk in text_chunks_progress
                ]
            self._service_context.llama_logger.add_log(
                {"summaries": summaries, "level": level}
            )

            event.on_end(payload={"summaries": summaries, "level": level})

        new_node_dict = self._construct_parent_nodes(
            index_graph, indices, cur_nodes_chunks, summaries
        )
        all_node_ids.update(new_node_dict)

        index_graph.root_nodes = new_node_dict

        if len(new_node_dict) <= self.num_children:
            return index_graph
        else:
            return self.build_index_from_nodes(
                index_graph, new_node_dict, all_node_ids, level=level + 1
            )

    async def abuild_index_from_nodes(
        self,
        index_graph: IndexGraph,
        cur_node_ids: Dict[int, str],
        all_node_ids: Dict[int, str],
        level: int = 0,
    ) -> IndexGraph:
        """Consolidates chunks recursively, in a bottoms-up fashion."""
        if len(cur_node_ids) <= self.num_children:
            index_graph.root_nodes = cur_node_ids
            return index_graph

        indices, cur_nodes_chunks, text_chunks = self._prepare_node_and_text_chunks(
            cur_node_ids
        )

        with self._service_context.callback_manager.event(
            CBEventType.TREE, payload={EventPayload.CHUNKS: text_chunks}
        ) as event:
            text_chunks_progress = get_tqdm_iterable(
                text_chunks,
                show_progress=self._show_progress,
                desc="Generating summaries",
            )
            tasks = [
                self._service_context.llm.apredict(
                    self.summary_prompt, context_str=text_chunk
                )
                for text_chunk in text_chunks_progress
            ]
            outputs: List[Tuple[str, str]] = await asyncio.gather(*tasks)
            summaries = [output[0] for output in outputs]
            self._service_context.llama_logger.add_log(
                {"summaries": summaries, "level": level}
            )

            event.on_end(payload={"summaries": summaries, "level": level})

        new_node_dict = self._construct_parent_nodes(
            index_graph, indices, cur_nodes_chunks, summaries
        )
        all_node_ids.update(new_node_dict)

        index_graph.root_nodes = new_node_dict

        if len(new_node_dict) <= self.num_children:
            return index_graph
        else:
            return await self.abuild_index_from_nodes(
                index_graph, new_node_dict, all_node_ids, level=level + 1
            )
