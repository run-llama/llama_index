from typing import Any, List, Optional, Sequence

from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.synthesizers.base import BaseSynthesizer
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.types import RESPONSE_TEXT_TYPE


class Generation(BaseSynthesizer):
    def __init__(
        self,
        simple_template: Optional[SimpleInputPrompt] = DEFAULT_SIMPLE_INPUT_PROMPT,
        service_context: Optional[ServiceContext] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming, node_postprocessors=node_postprocessors)
        self._input_prompt = simple_template

    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # NOTE: ignore text chunks and previous response
        del text_chunks

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
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # NOTE: ignore text chunks and previous response
        del text_chunks

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
