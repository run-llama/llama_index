"""Tests for the token counter."""

from types import SimpleNamespace

import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
from llama_index.core.utilities.token_counting import TokenCounter

TOOL_NAME = "get_weather"
TOOL_ARGS = '{"city": "Paris", "unit": "celsius"}'


def _legacy_tool_call() -> SimpleNamespace:
    """The shape ``from_openai_message`` stores in ``additional_kwargs``."""
    return SimpleNamespace(
        function=SimpleNamespace(name=TOOL_NAME, arguments=TOOL_ARGS)
    )


def _assistant_message(*, with_legacy_copy: bool) -> ChatMessage:
    """An assistant reply carrying one tool call, as converted from OpenAI."""
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_call_id="call_1", tool_name=TOOL_NAME, tool_kwargs=TOOL_ARGS
            )
        ],
        additional_kwargs=(
            {"tool_calls": [_legacy_tool_call()]} if with_legacy_copy else {}
        ),
    )


def test_estimate_tokens_counts_a_tool_call_once() -> None:
    """The backward-compatible copy is not counted on top of the block."""
    counter = TokenCounter(tokenizer=str.split)
    with_copy = counter.estimate_tokens_in_messages(
        [_assistant_message(with_legacy_copy=True)]
    )
    without_copy = counter.estimate_tokens_in_messages(
        [_assistant_message(with_legacy_copy=False)]
    )
    assert with_copy == without_copy


@pytest.mark.asyncio
async def test_aestimate_tokens_counts_a_tool_call_once() -> None:
    """The async twin behaves like the sync estimator."""
    counter = TokenCounter(tokenizer=str.split)
    with_copy = await counter.aestimate_tokens_in_messages(
        [_assistant_message(with_legacy_copy=True)]
    )
    without_copy = await counter.aestimate_tokens_in_messages(
        [_assistant_message(with_legacy_copy=False)]
    )
    assert with_copy == without_copy


def test_estimate_tokens_still_counts_legacy_tool_calls_without_blocks() -> None:
    """Messages that only carry the legacy copy keep being counted."""
    counter = TokenCounter(tokenizer=str.split)
    legacy_only = ChatMessage(
        role=MessageRole.ASSISTANT,
        additional_kwargs={"tool_calls": [_legacy_tool_call()]},
    )
    plain = ChatMessage(role=MessageRole.ASSISTANT)

    assert counter.estimate_tokens_in_messages([legacy_only]) > (
        counter.estimate_tokens_in_messages([plain])
    )
