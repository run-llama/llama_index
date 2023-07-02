from typing import List

import pytest

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.openai_utils import (from_openai_message_dicts,
                                           to_openai_message_dicts)


@pytest.fixture
def chat_messages_with_function_calling() -> List[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.USER, content="test question with functions"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": '{ "location": "Boston, MA"}',
                },
            },
        ),
        ChatMessage(
            role=MessageRole.FUNCTION,
            content='{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
            additional_kwargs={
                "name": "get_current_weather",
            },
        ),
    ]


@pytest.fixture
def openi_message_dicts_with_function_calling() -> List[dict]:
    return [
        {"role": "user", "content": "test question with functions"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": "get_current_weather",
                "arguments": '{ "location": "Boston, MA"}',
            },
        },
        {
            "role": "function",
            "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
            "name": "get_current_weather",
        },
    ]


def test_to_openai_message_dicts_basic_enum() -> None:
    chat_messages = [
        ChatMessage(role=MessageRole.USER, content="test question"),
        ChatMessage(role=MessageRole.ASSISTANT, content="test answer"),
    ]
    openai_messages = to_openai_message_dicts(chat_messages)
    assert openai_messages == [
        {"role": "user", "content": "test question"},
        {"role": "assistant", "content": "test answer"},
    ]


def test_to_openai_message_dicts_basic_string() -> None:
    chat_messages = [
        ChatMessage(role="user", content="test question"),
        ChatMessage(role="assistant", content="test answer"),
    ]
    openai_messages = to_openai_message_dicts(chat_messages)
    assert openai_messages == [
        {"role": "user", "content": "test question"},
        {"role": "assistant", "content": "test answer"},
    ]


def test_to_openai_message_dicts_function_calling(
    chat_messages_with_function_calling: List[ChatMessage],
    openi_message_dicts_with_function_calling: List[dict],
) -> None:
    openai_messages = to_openai_message_dicts(chat_messages_with_function_calling)
    assert openai_messages == openi_message_dicts_with_function_calling


def test_from_openai_message_dicts_function_calling(
    openi_message_dicts_with_function_calling: List[dict],
    chat_messages_with_function_calling: List[ChatMessage],
) -> None:
    chat_messages = from_openai_message_dicts(openi_message_dicts_with_function_calling)
    assert chat_messages == chat_messages_with_function_calling
