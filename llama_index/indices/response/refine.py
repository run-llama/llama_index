import logging
from typing import Any, Generator, Optional, Sequence, cast

from llama_index.indices.response.base_builder import BaseResponseBuilder
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.utils import truncate_text
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.response.utils import get_response_text
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.types import RESPONSE_TEXT_TYPE

logger = logging.getLogger(__name__)


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
        text_chunks = self._service_context.prompt_helper.repack(
            text_qa_template, [text_chunk]
        )

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
        text_chunks = self._service_context.prompt_helper.repack(
            refine_template, text_chunks=[text_chunk]
        )

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
            refine_template = self._refine_template.partial_format(
                query_str=query_str, existing_answer=response
            )

            self._log_prompt_and_response(
                formatted_prompt, response, log_prefix="Refined"
            )
        return response
