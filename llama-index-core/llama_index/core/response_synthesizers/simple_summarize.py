from typing import Any, Generator, Optional, Sequence, cast

from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.chat_prompts import CHAT_CONTENT_QA_PROMPT
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE


class SimpleSummarize(BaseSynthesizer):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        chat_content_qa_template: Optional[BasePromptTemplate] = None,
        streaming: bool = False,
        multimodal: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
            multimodal=multimodal,
        )
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
        self._chat_content_qa_template = (
            chat_content_qa_template or CHAT_CONTENT_QA_PROMPT
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "text_qa_template": self._text_qa_template,
            "chat_content_qa_template": self._chat_content_qa_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]
        if "chat_content_qa_template" in prompts:
            self._chat_content_qa_template = prompts["chat_content_qa_template"]

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        single_text_chunk = "\n".join(text_chunks)
        truncated_chunks = self._prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=[single_text_chunk],
            llm=self._llm,
        )

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            response = self._llm.predict(
                text_qa_template,
                context_str=truncated_chunks,
                **kwargs,
            )
        else:
            response = self._llm.stream(
                text_qa_template,
                context_str=truncated_chunks,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        single_text_chunk = "\n".join(text_chunks)
        truncated_chunks = self._prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=[single_text_chunk],
            llm=self._llm,
        )

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            response = await self._llm.apredict(
                text_qa_template,
                context_str=truncated_chunks,
                **response_kwargs,
            )
        else:
            response = await self._llm.astream(
                text_qa_template,
                context_str=truncated_chunks,
                **response_kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    def get_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        chat_content_qa_template = self._chat_content_qa_template.partial_format(
            query_str=query_str
        )
        single_message_chunk = ChatMessage.merge(
            splits=message_chunks,  # type: ignore[arg-type]
            chunk_size=sum(m.estimate_tokens() for m in message_chunks),  # type: ignore[union-attr, misc]
        )[0]
        truncated_chunks = self._prompt_helper.truncate(  # type: ignore[attr-defined, call-arg]
            prompt=chat_content_qa_template,
            messages=[single_message_chunk],
            llm=self._llm,
            strict=True,
        )

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            response = self._llm.predict(
                chat_content_qa_template,
                context_messages=truncated_chunks,
                **kwargs,
            )
        else:
            response = self._llm.stream(
                chat_content_qa_template,
                context_messages=truncated_chunks,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    async def aget_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        chat_content_qa_template = self._chat_content_qa_template.partial_format(
            query_str=query_str
        )
        single_message_chunk = ChatMessage.merge(
            splits=message_chunks,  # type: ignore[arg-type]
            chunk_size=sum(m.estimate_tokens() for m in message_chunks),  # type: ignore[union-attr, misc]
        )[0]
        truncated_chunks = await self._prompt_helper.atruncate(  # type: ignore[attr-defined]
            prompt=chat_content_qa_template,
            messages=[single_message_chunk],
            llm=self._llm,
        )

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            response = await self._llm.apredict(
                chat_content_qa_template,
                context_messages=truncated_chunks,
                **response_kwargs,
            )
        else:
            response = await self._llm.astream(
                chat_content_qa_template,
                context_messages=truncated_chunks,
                **response_kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response
