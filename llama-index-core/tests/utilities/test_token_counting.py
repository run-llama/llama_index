"""Tests for the TokenCounter message estimator."""

from types import SimpleNamespace

import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
from llama_index.core.utilities.token_counting import TokenCounter


def _word_tokenizer(text: str) -> list:
    """Deterministic tokenizer so token counts do not depend on tiktoken."""
    return text.split()


def _openai_style_tool_call() -> SimpleNamespace:
    """A tool call shaped like OpenAI's, exposing ``function.name``/``arguments``."""
    return SimpleNamespace(
        function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}')
    )


def _message_with_block_and_kwargs() -> ChatMessage:
    """Assistant message with the tool call in both a block and additional_kwargs.

    This mirrors how the OpenAI integration stores a native tool call: once as a
    ToolCallBlock and once in additional_kwargs for backward compatibility.
    """
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_call_id="call_1",
                tool_name="get_weather",
                tool_kwargs={"city": "Paris"},
            )
        ],
        additional_kwargs={"tool_calls": [_openai_style_tool_call()]},
    )


def _message_with_block_only() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ToolCallBlock(
                tool_call_id="call_1",
                tool_name="get_weather",
                tool_kwargs={"city": "Paris"},
            )
        ],
    )


def test_tool_call_in_block_and_kwargs_is_not_double_counted() -> None:
    counter = TokenCounter(tokenizer=_word_tokenizer)
    both = counter.estimate_tokens_in_messages([_message_with_block_and_kwargs()])
    block_only = counter.estimate_tokens_in_messages([_message_with_block_only()])
    assert both == block_only


def test_tool_call_only_in_additional_kwargs_is_still_counted() -> None:
    counter = TokenCounter(tokenizer=_word_tokenizer)
    kwargs_only = ChatMessage(
        role=MessageRole.ASSISTANT,
        additional_kwargs={"tool_calls": [_openai_style_tool_call()]},
    )
    empty = ChatMessage(role=MessageRole.ASSISTANT)
    assert counter.estimate_tokens_in_messages(
        [kwargs_only]
    ) > counter.estimate_tokens_in_messages([empty])


@pytest.mark.asyncio
async def test_async_tool_call_in_block_and_kwargs_is_not_double_counted() -> None:
    counter = TokenCounter(tokenizer=_word_tokenizer)
    both = await counter.aestimate_tokens_in_messages(
        [_message_with_block_and_kwargs()]
    )
    block_only = await counter.aestimate_tokens_in_messages(
        [_message_with_block_only()]
    )
    assert both == block_only
