from typing import Any, Sequence

from llama_index.prompts.mixin import PromptDictType
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.types import RESPONSE_TEXT_TYPE


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
