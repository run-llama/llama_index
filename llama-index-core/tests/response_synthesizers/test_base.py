"""Tests for the BaseSynthesizer abstract contract."""

from typing import Any, List

import pytest

from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE


class _MinimalSynthesizer(BaseSynthesizer):
    """
    Synthesizer that only implements the required abstract methods.

    Used to exercise the not-implemented contract on
    ``get_response_from_messages`` / ``aget_response_from_messages``.
    """

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def get_response(
        self,
        query_str: str,
        text_chunks: List[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        del query_str, text_chunks, response_kwargs
        return ""

    async def aget_response(
        self,
        query_str: str,
        text_chunks: List[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        del query_str, text_chunks, response_kwargs
        return ""


def test_get_response_from_messages_raises_descriptive() -> None:
    """The base class should raise NotImplementedError with a helpful message."""
    synth = _MinimalSynthesizer(llm=MockLLM())
    with pytest.raises(NotImplementedError, match="get_response_from_messages"):
        synth.get_response_from_messages("q", [])


@pytest.mark.asyncio
async def test_aget_response_from_messages_raises_descriptive() -> None:
    """The base class should raise NotImplementedError with a helpful message."""
    synth = _MinimalSynthesizer(llm=MockLLM())
    with pytest.raises(NotImplementedError, match="aget_response_from_messages"):
        await synth.aget_response_from_messages("q", [])
