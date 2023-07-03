from typing import Any, Optional, Sequence

from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.types import RESPONSE_TEXT_TYPE


class Generation(BaseSynthesizer):
    def __init__(
        self,
        simple_template: Optional[SimpleInputPrompt] = None,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming)
        self._input_prompt = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # NOTE: ignore text chunks and previous response
        del text_chunks

        if not self._streaming:
            response = await self._service_context.llm_predictor.apredict(
                self._input_prompt,
                query_str=query_str,
            )
            return response
        else:
            stream_response = self._service_context.llm_predictor.stream(
                self._input_prompt,
                query_str=query_str,
            )
            return stream_response

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # NOTE: ignore text chunks and previous response
        del text_chunks

        if not self._streaming:
            response = self._service_context.llm_predictor.predict(
                self._input_prompt,
                query_str=query_str,
            )
            return response
        else:
            stream_response = self._service_context.llm_predictor.stream(
                self._input_prompt,
                query_str=query_str,
            )
            return stream_response
