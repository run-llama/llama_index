from typing import Any, Generator, List, Optional, Sequence, cast

from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from llama_index.prompts.default_prompts import (
    DEFAULT_SIMPLE_INPUT_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.synthesizers.base import BaseSynthesizer
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.types import RESPONSE_TEXT_TYPE


class SimpleSummarize(BaseSynthesizer):
    def __init__(
        self,
        text_qa_template: QuestionAnswerPrompt = DEFAULT_TEXT_QA_PROMPT,
        service_context: Optional[ServiceContext] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming, node_postprocessors=node_postprocessors)
        self._text_qa_template = text_qa_template

    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        truncated_chunks = self._service_context.prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=text_chunks,
        )
        node_text = "\n".join(truncated_chunks)

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            (
                response,
                formatted_prompt,
            ) = await self._service_context.llm_predictor.apredict(
                text_qa_template,
                context_str=node_text,
            )
        else:
            response, formatted_prompt = self._service_context.llm_predictor.stream(
                text_qa_template,
                context_str=node_text,
            )
        self._log_prompt_and_response(formatted_prompt, response)

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
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        truncated_chunks = self._service_context.prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=text_chunks,
        )
        node_text = "\n".join(truncated_chunks)

        response: RESPONSE_TEXT_TYPE
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

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response
