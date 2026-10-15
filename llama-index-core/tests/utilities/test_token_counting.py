import asyncio
from typing import Any
from types import SimpleNamespace
import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.utilities.token_counting import TokenCounter


def dummy_tokenizer(text: str) -> list[str]:
    return text.split()


def test_estimate_tokens_in_messages_dict_tool_calls():
    counter = TokenCounter(tokenizer=dummy_tokenizer)

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "Hanoi Vietnam",
                    },
                }
            ]
        },
    )

    tokens = counter.estimate_tokens_in_messages([msg])
    # Role 'assistant' (1) + content 'calling tool' (2) + func_name 'get_weather' (1) + args 'Hanoi Vietnam' (2) + tool_call overhead (3) + msg overhead (3) = 12
    assert tokens == 12


def test_estimate_tokens_in_messages_object_tool_calls():
    counter = TokenCounter(tokenizer=dummy_tokenizer)

    tool_call_obj = SimpleNamespace(
        function=SimpleNamespace(name="get_weather", arguments="Hanoi Vietnam")
    )

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={"tool_calls": [tool_call_obj]},
    )

    tokens = counter.estimate_tokens_in_messages([msg])
    assert tokens == 12


def test_estimate_tokens_in_messages_none_guards():
    counter = TokenCounter(tokenizer=dummy_tokenizer)

    # Tool call with None arguments or empty dict
    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "arguments": None,
                    },
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": None,
                },
            ]
        },
    )

    tokens = counter.estimate_tokens_in_messages([msg])
    # Should not raise TypeError and count properly
    assert tokens > 0


def test_estimate_tokens_in_messages_conservation_invariant():
    """Drift guard: ensure TokenCounter estimate stays within sane conservation band against mock provider usage."""
    counter = TokenCounter(tokenizer=dummy_tokenizer)

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="I will query the database now.",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_101",
                    "type": "function",
                    "function": {
                        "name": "sql_query",
                        "arguments": '{"query": "SELECT * FROM users WHERE active = 1"}',
                    },
                }
            ]
        },
    )

    estimated_tokens = counter.estimate_tokens_in_messages([msg])
    raw_content_tokens = len(str(msg.content or "").split())
    raw_tool_tokens = len("sql_query".split()) + len('{"query": "SELECT * FROM users WHERE active = 1"}'.split())
    expected_tokens = raw_content_tokens + raw_tool_tokens

    assert estimated_tokens == expected_tokens
    assert 0.85 * expected_tokens <= estimated_tokens <= 1.15 * expected_tokens



@pytest.mark.asyncio
async def test_aestimate_tokens_in_messages_dict_tool_calls():
    counter = TokenCounter(tokenizer=dummy_tokenizer)

    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "Hanoi Vietnam",
                    },
                }
            ]
        },
    )

    tokens = await counter.aestimate_tokens_in_messages([msg])
    assert tokens == 12
