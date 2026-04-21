from typing import Any, Sequence

from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE


class ContextOnly(BaseSynthesizer):
    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return "\n\n".join(text_chunks)

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return "\n\n".join(text_chunks)

    def get_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        texts = [
            block.text
            for msg in message_chunks
            for block in msg.blocks
            if block.block_type == "text"
        ]
        return "\n\n".join(texts)

    async def aget_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        texts = [
            block.text
            for msg in message_chunks
            for block in msg.blocks
            if block.block_type == "text"
        ]
        return "\n\n".join(texts)
