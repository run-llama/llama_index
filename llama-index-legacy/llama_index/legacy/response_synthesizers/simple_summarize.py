from typing import Any, Generator, Optional, Sequence, cast

from llama_index.legacy.prompts import BasePromptTemplate
from llama_index.legacy.prompts.default_prompt_selectors import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.legacy.prompts.mixin import PromptDictType
from llama_index.legacy.response_synthesizers.base import BaseSynthesizer
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.types import RESPONSE_TEXT_TYPE


class SimpleSummarize(BaseSynthesizer):
    def __init__(
        self,
        text_qa_template: Optional[BasePromptTemplate] = None,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming)
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"text_qa_template": self._text_qa_template}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]

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
            response = await self._service_context.llm.apredict(
                text_qa_template,
                context_str=node_text,
                **response_kwargs,
            )
        else:
            response = self._service_context.llm.stream(
                text_qa_template,
                context_str=node_text,
                **response_kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

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
            response = self._service_context.llm.predict(
                text_qa_template,
                context_str=node_text,
                **kwargs,
            )
        else:
            response = self._service_context.llm.stream(
                text_qa_template,
                context_str=node_text,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response
