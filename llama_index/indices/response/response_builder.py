"""Response builder class.

This class provides general functions for taking in a set of text
and generating a response.

Will support different modes, from 1) stuffing chunks into prompt,
2) create and refine separately over each chunk, 3) tree summarization.

"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, cast

from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.data_structs import IndexGraph
from llama_index.data_structs.node import Node
from llama_index.storage.docstore.registry import get_default_docstore
from llama_index.indices.common_tree.base import GPTTreeIndexBuilder
from llama_index.indices.response.type import ResponseMode
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.utils import get_sorted_node_list, truncate_text
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from llama_index.prompts.default_prompts import (
    DEFAULT_SIMPLE_INPUT_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
    SummaryPrompt,
)
from llama_index.response.utils import get_response_text
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.types import RESPONSE_TEXT_TYPE
from llama_index.utils import temp_set_attrs

logger = logging.getLogger(__name__)


class BaseResponseBuilder(ABC):
    """Response builder class."""

    def __init__(
        self,
        service_context: ServiceContext,
        streaming: bool = False,
    ) -> None:
        """Init params."""
        self._service_context = service_context
        self._streaming = streaming

    @property
    def service_context(self) -> ServiceContext:
        return self._service_context

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

    def _callback_llm_on_start(self) -> str:
        """Call the callback manager on_start and return event id."""
        event_id = self._service_context.callback_manager.on_event_start(
            CBEventType.LLM
        )
        return event_id

    def _callback_llm_on_end(
        self,
        formatted_prompt: str,
        response: RESPONSE_TEXT_TYPE,
        event_id: str,
        stage: Optional[str] = "",
    ) -> None:
        self._service_context.callback_manager.on_event_end(
            CBEventType.LLM,
            payload={
                "stage": stage,
                "response": response,
                "formatted_prompt": formatted_prompt,
            },
            event_id=event_id,
        )

    def _callback_chunking_on_start(self, text: str) -> str:
        event_id = self._service_context.callback_manager.on_event_start(
            CBEventType.CHUNKING, payload={"node": text}
        )
        return event_id

    def _callback_chunking_on_end(
        self, text_chunks: Sequence[str], event_id: str
    ) -> None:
        self._service_context.callback_manager.on_event_end(
            CBEventType.CHUNKING, payload={"chunks": text_chunks}, event_id=event_id
        )

    @abstractmethod
    @llm_token_counter("get_response")
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...

    @abstractmethod
    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...


class Refine(BaseResponseBuilder):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        refine_template: RefinePrompt,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming)
        self.text_qa_template = text_qa_template
        self._refine_template = refine_template

    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return self.get_response(query_str, text_chunks, prev_response)

    @llm_token_counter("get_response")
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response over chunks."""
        prev_response_obj = cast(Optional[RESPONSE_TEXT_TYPE], prev_response)
        response: Optional[RESPONSE_TEXT_TYPE] = None
        for text_chunk in text_chunks:
            if prev_response_obj is None:
                # if this is the first chunk, and text chunk already
                # is an answer, then return it
                response = self._give_response_single(
                    query_str,
                    text_chunk,
                )
            else:
                response = self._refine_response_single(
                    prev_response_obj, query_str, text_chunk
                )
            prev_response_obj = response
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    def _give_response_single(
        self,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response given a query and a corresponding text chunk."""
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        qa_text_splitter = (
            self._service_context.prompt_helper.get_text_splitter_given_prompt(
                text_qa_template, 1
            )
        )
        event_id = self._callback_chunking_on_start(text_chunk)
        text_chunks = qa_text_splitter.split_text(text_chunk)
        self._callback_chunking_on_end(text_chunk, event_id)

        response: Optional[RESPONSE_TEXT_TYPE] = None
        # TODO: consolidate with loop in get_response_default
        for cur_text_chunk in text_chunks:
            if response is None and not self._streaming:
                event_id = self._callback_llm_on_start()
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
                self._callback_llm_on_end(
                    formatted_prompt, response, event_id, stage="Initial"
                )
            elif response is None and self._streaming:
                event_id = self._callback_llm_on_start()
                response, formatted_prompt = self._service_context.llm_predictor.stream(
                    text_qa_template,
                    context_str=cur_text_chunk,
                )
                self._log_prompt_and_response(
                    formatted_prompt, response, log_prefix="Initial"
                )
                self._callback_llm_on_end(
                    formatted_prompt, response, event_id, stage="Initial"
                )
            else:
                response = self._refine_response_single(
                    cast(RESPONSE_TEXT_TYPE, response),
                    query_str,
                    cur_text_chunk,
                )
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    def _refine_response_single(
        self,
        response: RESPONSE_TEXT_TYPE,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Refine response."""
        # TODO: consolidate with logic in response/schema.py
        if isinstance(response, Generator):
            response = get_response_text(response)

        fmt_text_chunk = truncate_text(text_chunk, 50)
        logger.debug(f"> Refine context: {fmt_text_chunk}")
        # NOTE: partial format refine template with query_str and existing_answer here
        refine_template = self._refine_template.partial_format(
            query_str=query_str, existing_answer=response
        )
        refine_text_splitter = (
            self._service_context.prompt_helper.get_text_splitter_given_prompt(
                refine_template, 1
            )
        )

        event_id = self._callback_chunking_on_start(text_chunk)
        text_chunks = refine_text_splitter.split_text(text_chunk)
        self._callback_chunking_on_end(text_chunks, event_id)

        for cur_text_chunk in text_chunks:
            event_id = self._callback_llm_on_start()
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
            refine_template = self._refine_template.partial_format(
                query_str=query_str, existing_answer=response
            )

            self._log_prompt_and_response(
                formatted_prompt, response, log_prefix="Refined"
            )
            self._callback_llm_on_end(
                formatted_prompt, response, event_id, stage="Refined"
            )
        return response


class CompactAndRefine(Refine):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        refine_template: RefinePrompt,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return self.get_response(query_str, text_chunks, prev_response)

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get compact response."""
        # use prompt helper to fix compact text_chunks under the prompt limitation
        # TODO: This is a temporary fix - reason it's temporary is that
        # the refine template does not account for size of previous answer.
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        refine_template = self._refine_template.partial_format(query_str=query_str)

        max_prompt = self._service_context.prompt_helper.get_biggest_prompt(
            [text_qa_template, refine_template]
        )
        with temp_set_attrs(
            self._service_context.prompt_helper, use_chunk_size_limit=False
        ):
            new_texts = self._service_context.prompt_helper.compact_text_chunks(
                max_prompt, text_chunks
            )
            response = super().get_response(
                query_str=query_str, text_chunks=new_texts, prev_response=prev_response
            )
        return response


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
        summary_template = SummaryPrompt.from_prompt(text_qa_template)

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
        # first join all the text chunks into a single text
        all_text = "\n\n".join(text_chunks)

        # then get text splitter
        text_splitter = (
            self._service_context.prompt_helper.get_text_splitter_given_prompt(
                summary_template, num_children
            )
        )
        event_id = self._callback_chunking_on_start(all_text)
        text_chunks = text_splitter.split_text(all_text)
        self._callback_chunking_on_end(text_chunks, event_id)

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
        node_text = self._service_context.prompt_helper.get_text_from_nodes(
            node_list, prompt=text_qa_template
        )
        # NOTE: the final response could be a string or a stream
        response = super().get_response(
            query_str=query_str,
            text_chunks=[node_text],
            prev_response=prev_response,
        )
        if isinstance(response, str):
            response = response or "Empty Response"
        return response


class SimpleSummarize(BaseResponseBuilder):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context, streaming)
        self._text_qa_template = text_qa_template

    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        node_text = self._service_context.prompt_helper.get_text_from_nodes(
            [Node(text=text) for text in text_chunks], prompt=text_qa_template
        )

        response: RESPONSE_TEXT_TYPE
        event_id = self._callback_llm_on_start()
        if not self._streaming:
            (
                response,
                formatted_prompt,
            ) = await self._service_context.llm_predictor.apredict(
                text_qa_template,
                context_str=node_text,
            )
        else:
            event_id = self._callback_llm_on_start()
            response, formatted_prompt = self._service_context.llm_predictor.stream(
                text_qa_template,
                context_str=node_text,
            )
        self._log_prompt_and_response(formatted_prompt, response)
        self._callback_llm_on_end(
            formatted_prompt, response, event_id, stage="summarize"
        )
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    @llm_token_counter("get_response")
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        node_text = self._service_context.prompt_helper.get_text_from_nodes(
            [Node(text=text) for text in text_chunks], prompt=text_qa_template
        )

        response: RESPONSE_TEXT_TYPE
        event_id = self._callback_llm_on_start()
        if not self._streaming:
            (response, formatted_prompt,) = self._service_context.llm_predictor.predict(
                text_qa_template,
                context_str=node_text,
            )
        else:
            response, formatted_prompt = self._service_context.llm_predictor.stream(
                text_qa_template,
                context_str=node_text,
            )
        self._log_prompt_and_response(formatted_prompt, response)
        self._callback_llm_on_end(
            formatted_prompt, response, event_id, stage="summarize"
        )
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response


class Generation(BaseResponseBuilder):
    def __init__(
        self,
        service_context: ServiceContext,
        simple_template: Optional[SimpleInputPrompt] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context, streaming)
        self._input_prompt = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT

    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # NOTE: ignore text chunks and previous response
        del text_chunks
        del prev_response

        if not self._streaming:
            event_id = self._callback_llm_on_start()
            (
                response,
                formatted_prompt,
            ) = await self._service_context.llm_predictor.apredict(
                self._input_prompt,
                query_str=query_str,
            )
            self._callback_llm_on_end(formatted_prompt, response, event_id)
            return response
        else:
            stream_response, _ = self._service_context.llm_predictor.stream(
                self._input_prompt,
                query_str=query_str,
            )
            return stream_response

    @llm_token_counter("get_response")
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # NOTE: ignore text chunks and previous response
        del text_chunks
        del prev_response

        if not self._streaming:
            event_id = self._callback_llm_on_start()
            response, formatted_prompt = self._service_context.llm_predictor.predict(
                self._input_prompt,
                query_str=query_str,
            )
            self._callback_llm_on_end(formatted_prompt, response, event_id)
            return response
        else:
            stream_response, _ = self._service_context.llm_predictor.stream(
                self._input_prompt,
                query_str=query_str,
            )
            return stream_response


def get_response_builder(
    service_context: ServiceContext,
    text_qa_template: Optional[QuestionAnswerPrompt] = None,
    refine_template: Optional[RefinePrompt] = None,
    simple_template: Optional[SimpleInputPrompt] = None,
    mode: ResponseMode = ResponseMode.COMPACT,
    use_async: bool = False,
    streaming: bool = False,
) -> BaseResponseBuilder:
    text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
    refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
    simple_template = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT
    if mode == ResponseMode.REFINE:
        return Refine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )
    elif mode == ResponseMode.COMPACT:
        return CompactAndRefine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )
    elif mode == ResponseMode.TREE_SUMMARIZE:
        return TreeSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
            use_async=use_async,
        )
    elif mode == ResponseMode.SIMPLE_SUMMARIZE:
        return SimpleSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
        )
    elif mode == ResponseMode.GENERATION:
        return Generation(
            service_context=service_context,
            simple_template=simple_template,
            streaming=streaming,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
