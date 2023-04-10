"""Response builder class.

This class provides general functions for taking in a set of text
and generating a response.

Will support different modes, from 1) stuffing chunks into prompt,
2) create and refine separately over each chunk, 3) tree summarization.

"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

from gpt_index.data_structs.data_structs_v2 import IndexGraph
from gpt_index.data_structs.node_v2 import Node, NodeWithScore
from gpt_index.docstore import DocumentStore
from gpt_index.indices.common_tree.base import GPTTreeIndexBuilder
from gpt_index.indices.service_context import ServiceContext
from gpt_index.indices.utils import get_sorted_node_list, truncate_text
from gpt_index.logger.base import LlamaLogger
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt, SummaryPrompt
from gpt_index.response.utils import get_response_text
from gpt_index.types import RESPONSE_TEXT_TYPE
from gpt_index.utils import temp_set_attrs

logger = logging.getLogger(__name__)


class ResponseMode(str, Enum):
    """Response modes."""

    DEFAULT = "default"
    COMPACT = "compact"
    TREE_SUMMARIZE = "tree_summarize"
    NO_TEXT = "no_text"


@dataclass
class TextChunk:
    """Response chunk."""

    text: str
    # Whether this chunk is already a response
    is_answer: bool = False


class ResponseBuilder:
    """Response builder class."""

    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        refine_template: RefinePrompt,
        texts: Optional[List[TextChunk]] = None,
        nodes: Optional[List[Node]] = None,
        use_async: bool = False,
        streaming: bool = False,
    ) -> None:
        """Init params."""
        self._service_context = service_context
        self.text_qa_template = text_qa_template
        self.refine_template = refine_template
        self._texts = texts or []
        nodes = nodes or []
        self.source_nodes: List[NodeWithScore] = [NodeWithScore(node) for node in nodes]
        self._use_async = use_async
        self._streaming = streaming

    def _log_prompt_and_response(
        self,
        formatted_prompt: str,
        response: RESPONSE_TEXT_TYPE,
        log_prefix: str = "",
    ) -> None:
        """Log prompt and response from LLM."""
        logger.debug(f"> {log_prefix} prompt template: {formatted_prompt}")
        self._service_context.llama_logger.add_log(
            {"formatted_prompt_template": formatted_prompt}
        )
        logger.debug(f"> {log_prefix} response: {response}")
        self._service_context.llama_logger.add_log(
            {f"{log_prefix.lower()}_response": response or "Empty Response"}
        )

    def add_text_chunks(self, text_chunks: List[TextChunk]) -> None:
        """Add text chunk."""
        self._texts.extend(text_chunks)

    def reset(self) -> None:
        """Clear text chunks."""
        self._texts = []
        self.source_nodes = []

    def add_node_as_source(
        self, node: Node, similarity: Optional[float] = None
    ) -> None:
        """Add node."""
        self.add_node_with_score(NodeWithScore(node=node, score=similarity))

    def add_node_with_score(self, node_with_score: NodeWithScore) -> None:
        """Add source node directly."""
        self.source_nodes.append(node_with_score)

    def get_sources(self) -> List[NodeWithScore]:
        """Get sources."""
        return self.source_nodes

    def get_logger(self) -> LlamaLogger:
        """Get logger."""
        return self._service_context.llama_logger

    def refine_response_single(
        self,
        response: RESPONSE_TEXT_TYPE,
        query_str: str,
        text_chunk: str,
    ) -> RESPONSE_TEXT_TYPE:
        """Refine response."""
        # TODO: consolidate with logic in response/schema.py
        if isinstance(response, Generator):
            response = get_response_text(response)

        fmt_text_chunk = truncate_text(text_chunk, 50)
        logger.debug(f"> Refine context: {fmt_text_chunk}")
        # NOTE: partial format refine template with query_str and existing_answer here
        refine_template = self.refine_template.partial_format(
            query_str=query_str, existing_answer=response
        )
        refine_text_splitter = (
            self._service_context.prompt_helper.get_text_splitter_given_prompt(
                refine_template, 1
            )
        )
        text_chunks = refine_text_splitter.split_text(text_chunk)
        for cur_text_chunk in text_chunks:
            if not self._streaming:
                (
                    response,
                    formatted_prompt,
                ) = self._service_context.llm_predictor.predict(
                    refine_template,
                    context_msg=cur_text_chunk,
                )
            else:
                response, formatted_prompt = self._service_context.llm_predictor.stream(
                    refine_template,
                    context_msg=cur_text_chunk,
                )
            refine_template = self.refine_template.partial_format(
                query_str=query_str, existing_answer=response
            )

            self._log_prompt_and_response(
                formatted_prompt, response, log_prefix="Refined"
            )
        return response

    def give_response_single(
        self,
        query_str: str,
        text_chunk: str,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response given a query and a corresponding text chunk."""
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        qa_text_splitter = (
            self._service_context.prompt_helper.get_text_splitter_given_prompt(
                text_qa_template, 1
            )
        )
        text_chunks = qa_text_splitter.split_text(text_chunk)
        response: Optional[RESPONSE_TEXT_TYPE] = None
        # TODO: consolidate with loop in get_response_default
        for cur_text_chunk in text_chunks:
            if response is None and not self._streaming:
                (
                    response,
                    formatted_prompt,
                ) = self._service_context.llm_predictor.predict(
                    text_qa_template,
                    context_str=cur_text_chunk,
                )
                self._log_prompt_and_response(
                    formatted_prompt, response, log_prefix="Initial"
                )
            elif response is None and self._streaming:
                response, formatted_prompt = self._service_context.llm_predictor.stream(
                    text_qa_template,
                    context_str=cur_text_chunk,
                )
                self._log_prompt_and_response(
                    formatted_prompt, response, log_prefix="Initial"
                )
            else:
                response = self.refine_response_single(
                    cast(RESPONSE_TEXT_TYPE, response),
                    query_str,
                    cur_text_chunk,
                )
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    def get_response_over_chunks(
        self,
        query_str: str,
        text_chunks: List[TextChunk],
        prev_response: Optional[str] = None,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response over chunks."""
        prev_response_obj = cast(Optional[RESPONSE_TEXT_TYPE], prev_response)
        response: Optional[RESPONSE_TEXT_TYPE] = None
        for text_chunk in text_chunks:
            if prev_response_obj is None:
                # if this is the first chunk, and text chunk already
                # is an answer, then return it
                if text_chunk.is_answer:
                    response = text_chunk.text
                # otherwise give response
                else:
                    response = self.give_response_single(
                        query_str,
                        text_chunk.text,
                    )
            else:
                response = self.refine_response_single(
                    prev_response_obj, query_str, text_chunk.text
                )
            prev_response_obj = response
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    def _get_response_default(
        self, query_str: str, prev_response: Optional[str]
    ) -> RESPONSE_TEXT_TYPE:
        return self.get_response_over_chunks(
            query_str, self._texts, prev_response=prev_response
        )

    def _get_response_compact(
        self, query_str: str, prev_response: Optional[str]
    ) -> RESPONSE_TEXT_TYPE:
        """Get compact response."""
        # use prompt helper to fix compact text_chunks under the prompt limitation
        # TODO: This is a temporary fix - reason it's temporary is that
        # the refine template does not account for size of previous answer.
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        refine_template = self.refine_template.partial_format(query_str=query_str)

        max_prompt = self._service_context.prompt_helper.get_biggest_prompt(
            [text_qa_template, refine_template]
        )
        with temp_set_attrs(
            self._service_context.prompt_helper, use_chunk_size_limit=False
        ):
            new_texts = self._service_context.prompt_helper.compact_text_chunks(
                max_prompt, [t.text for t in self._texts]
            )
            new_text_chunks = [TextChunk(text=t) for t in new_texts]
            response = self.get_response_over_chunks(
                query_str, new_text_chunks, prev_response=prev_response
            )
        return response

    def _get_tree_index_builder_and_nodes(
        self,
        summary_template: SummaryPrompt,
        query_str: str,
        num_children: int = 10,
    ) -> Tuple[GPTTreeIndexBuilder, List[Node]]:
        """Get tree index builder."""
        # first join all the text chunks into a single text
        all_text = "\n\n".join([t.text for t in self._texts])
        # then get text splitter
        text_splitter = (
            self._service_context.prompt_helper.get_text_splitter_given_prompt(
                summary_template, num_children
            )
        )
        text_chunks = text_splitter.split_text(all_text)
        nodes = [Node(text=t) for t in text_chunks]

        docstore = DocumentStore()
        docstore.add_documents(nodes, allow_update=False)
        index_builder = GPTTreeIndexBuilder(
            num_children,
            summary_template,
            service_context=self._service_context,
            docstore=docstore,
            use_async=self._use_async,
        )
        return index_builder, nodes

    def _get_tree_response_over_root_nodes(
        self,
        query_str: str,
        prev_response: Optional[str],
        root_nodes: Dict[int, Node],
        text_qa_template: QuestionAnswerPrompt,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response from tree builder over root nodes."""
        node_list = get_sorted_node_list(root_nodes)
        node_text = self._service_context.prompt_helper.get_text_from_nodes(
            node_list, prompt=text_qa_template
        )
        # NOTE: the final response could be a string or a stream
        response = self.get_response_over_chunks(
            query_str,
            [TextChunk(node_text)],
            prev_response=prev_response,
        )
        if isinstance(response, str):
            response = response or "Empty Response"
        return response

    def _get_response_tree_summarize(
        self,
        query_str: str,
        prev_response: Optional[str],
        num_children: int = 10,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        summary_template = SummaryPrompt.from_prompt(text_qa_template)

        index_builder, nodes = self._get_tree_index_builder_and_nodes(
            summary_template, query_str, num_children
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

    async def _aget_response_tree_summarize(
        self,
        query_str: str,
        prev_response: Optional[str],
        num_children: int = 10,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        summary_template = SummaryPrompt.from_prompt(text_qa_template)

        index_builder, nodes = self._get_tree_index_builder_and_nodes(
            summary_template, query_str, num_children
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

    def get_response(
        self,
        query_str: str,
        prev_response: Optional[str] = None,
        mode: ResponseMode = ResponseMode.DEFAULT,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        if mode == ResponseMode.DEFAULT:
            return self._get_response_default(query_str, prev_response)
        elif mode == ResponseMode.COMPACT:
            return self._get_response_compact(query_str, prev_response)
        elif mode == ResponseMode.TREE_SUMMARIZE:
            return self._get_response_tree_summarize(
                query_str, prev_response, **response_kwargs
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    async def aget_response(
        self,
        query_str: str,
        prev_response: Optional[str] = None,
        mode: ResponseMode = ResponseMode.DEFAULT,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        # NOTE: for default and compact response modes, return synchronous version
        if mode == ResponseMode.DEFAULT:
            return self._get_response_default(query_str, prev_response)
        elif mode == ResponseMode.COMPACT:
            return self._get_response_compact(query_str, prev_response)
        elif mode == ResponseMode.TREE_SUMMARIZE:
            return await self._aget_response_tree_summarize(
                query_str, prev_response, **response_kwargs
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
