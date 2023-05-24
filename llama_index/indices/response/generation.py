from typing import Any, Optional, Sequence

from llama_index.indices.response.base_builder import BaseResponseBuilder
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.types import RESPONSE_TEXT_TYPE


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
            (
                response,
                formatted_prompt,
            ) = await self._service_context.llm_predictor.apredict(
                self._input_prompt,
                query_str=query_str,
            )
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
            response, formatted_prompt = self._service_context.llm_predictor.predict(
                self._input_prompt,
                query_str=query_str,
            )
            return response
        else:
            stream_response, _ = self._service_context.llm_predictor.stream(
                self._input_prompt,
                query_str=query_str,
            )
            return stream_response
