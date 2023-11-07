from typing import List

import pytest
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.openai_utils import (
    from_openai_messages,
    to_openai_message_dicts,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)


@pytest.fixture()
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


@pytest.fixture()
def openi_message_dicts_with_function_calling() -> List[ChatCompletionMessage]:
    return [
        ChatCompletionMessage(role="user", content="test question with functions"),
        ChatCompletionMessage(
            role="assistant",
            content=None,
            function_call=FunctionCall(
                name="get_current_weather",
                arguments='{ "location": "Boston, MA"}',
            ),
        ),
        ChatCompletionMessage(
            role="function",
            content='{"temperature": "22", "unit": "celsius", '
            '"description": "Sunny"}',
            name="get_current_weather",
        ),
    ]


@pytest.fixture()
def azure_openi_message_dicts_with_function_calling() -> List[ChatCompletionMessage]:
    """
    Taken from:
    - https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling.
    """
    return [
        ChatCompletionMessage(
            role="assistant",
            content=None,
            function_call=FunctionCall(
                name="search_hotels",
                arguments='{\n  "location": "San Diego",\n  "max_price": 300,\n  "features": "beachfront,free breakfast"\n}',
            ),
        )
    ]


@pytest.fixture()
def azure_chat_messages_with_function_calling() -> List[ChatMessage]:
    return [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={
                "function_call": {
                    "name": "search_hotels",
                    "arguments": '{\n  "location": "San Diego",\n  "max_price": 300,\n  "features": "beachfront,free breakfast"\n}',
                },
            },
        ),
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
    openi_message_dicts_with_function_calling: List[ChatCompletionMessage],
) -> None:
    openai_messages = to_openai_message_dicts(chat_messages_with_function_calling)
    for openai_message, openai_message_dict in zip(
        openai_messages, openi_message_dicts_with_function_calling
    ):
        for key in openai_message:
            assert openai_message[key] == getattr(openai_message_dict, key)


def test_from_openai_message_dicts_function_calling(
    openi_message_dicts_with_function_calling: List[ChatCompletionMessage],
    chat_messages_with_function_calling: List[ChatMessage],
) -> None:
    chat_messages = from_openai_messages(openi_message_dicts_with_function_calling)

    print(chat_messages)
    print(chat_messages_with_function_calling)

    # assert attributes match
    for chat_message, chat_message_with_function_calling in zip(
        chat_messages, chat_messages_with_function_calling
    ):
        for key in chat_message.additional_kwargs:
            assert chat_message.additional_kwargs[
                key
            ] == chat_message_with_function_calling.additional_kwargs.get(key, None)
        assert chat_message.content == chat_message_with_function_calling.content
        assert chat_message.role == chat_message_with_function_calling.role


def test_from_openai_message_dicts_function_calling_azure(
    azure_openi_message_dicts_with_function_calling: List[ChatCompletionMessage],
    azure_chat_messages_with_function_calling: List[ChatMessage],
) -> None:
    chat_messages = from_openai_messages(
        azure_openi_message_dicts_with_function_calling
    )
    assert chat_messages == azure_chat_messages_with_function_calling
