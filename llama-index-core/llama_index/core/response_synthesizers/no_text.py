from typing import Any, Sequence

from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import (
    BaseSynthesizer,
    BaseMultimodalSynthesizer,
)
from llama_index.core.types import RESPONSE_TEXT_TYPE


class NoText(BaseSynthesizer):
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
        return ""

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return ""


class MultimodalNoText(BaseMultimodalSynthesizer):
    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def get_response(  # type: ignore[override]
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return ""

    async def aget_response(  # type: ignore[override]
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return ""
