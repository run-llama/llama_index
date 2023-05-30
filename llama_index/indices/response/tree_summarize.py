from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.data_structs.data_structs import IndexGraph
from llama_index.data_structs.node import Node
from llama_index.indices.common_tree.base import GPTTreeIndexBuilder
from llama_index.indices.response.refine import Refine
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.utils import get_sorted_node_list
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SummaryPrompt,
)
from llama_index.storage.docstore.registry import get_default_docstore
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.types import RESPONSE_TEXT_TYPE


class TreeSummarize(Refine):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        refine_template: RefinePrompt,
        streaming: bool = False,
        use_async: bool = True,
    ) -> None:
        super().__init__(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )
        self._use_async = use_async

    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        num_children: int = 10,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        summary_template = SummaryPrompt.from_prompt(text_qa_template)

        index_builder, nodes = self._get_tree_index_builder_and_nodes(
            summary_template, query_str, text_chunks, num_children
        )
        index_graph = IndexGraph()
        for node in nodes:
            index_graph.insert(node)
        index_graph = await index_builder.abuild_index_from_nodes(
            index_graph, index_graph.all_nodes, index_graph.all_nodes
        )
        root_node_ids = index_graph.root_nodes
        root_nodes = {
            index: index_builder.docstore.get_node(node_id)
            for index, node_id in root_node_ids.items()
        }
        return self._get_tree_response_over_root_nodes(
            query_str, prev_response, root_nodes, text_qa_template
        )

    @llm_token_counter("get_response")
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        num_children: int = 10,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        summary_template = SummaryPrompt.from_prompt(
            text_qa_template, prompt_type=PromptType.SUMMARY
        )

        index_builder, nodes = self._get_tree_index_builder_and_nodes(
            summary_template, query_str, text_chunks, num_children
        )
        index_graph = IndexGraph()
        for node in nodes:
            index_graph.insert(node)
        index_graph = index_builder.build_index_from_nodes(
            index_graph, index_graph.all_nodes, index_graph.all_nodes
        )
        root_node_ids = index_graph.root_nodes
        root_nodes = {
            index: index_builder.docstore.get_node(node_id)
            for index, node_id in root_node_ids.items()
        }
        return self._get_tree_response_over_root_nodes(
            query_str, prev_response, root_nodes, text_qa_template
        )

    def _get_tree_index_builder_and_nodes(
        self,
        summary_template: SummaryPrompt,
        query_str: str,
        text_chunks: Sequence[str],
        num_children: int = 10,
    ) -> Tuple[GPTTreeIndexBuilder, List[Node]]:
        """Get tree index builder."""
        text_chunks = self._service_context.prompt_helper.repack(
            summary_template, text_chunks=text_chunks
        )
        new_nodes = [Node(text=t) for t in text_chunks]

        docstore = get_default_docstore()
        docstore.add_documents(new_nodes, allow_update=False)
        index_builder = GPTTreeIndexBuilder(
            num_children,
            summary_template,
            service_context=self._service_context,
            docstore=docstore,
            use_async=self._use_async,
        )
        return index_builder, new_nodes

    def _get_tree_response_over_root_nodes(
        self,
        query_str: str,
        prev_response: Optional[str],
        root_nodes: Dict[int, Node],
        text_qa_template: QuestionAnswerPrompt,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response from tree builder over root text_chunks."""
        node_list = get_sorted_node_list(root_nodes)
        truncated_chunks = self._service_context.prompt_helper.truncate(
            prompt=text_qa_template, text_chunks=[node.get_text() for node in node_list]
        )
        node_text = "\n".join(truncated_chunks)
        # NOTE: the final response could be a string or a stream
        response = super().get_response(
            query_str=query_str,
            text_chunks=[node_text],
            prev_response=prev_response,
        )
        if isinstance(response, str):
            response = response or "Empty Response"
        return response
