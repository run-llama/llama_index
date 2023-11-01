from typing import List

import openai
import pytest
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.openai_utils import (
    create_retry_decorator,
    from_openai_message_dicts,
    to_openai_message_dicts,
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
            "content": '{"temperature": "22", "unit": "celsius", '
            '"description": "Sunny"}',
            "name": "get_current_weather",
        },
    ]


@pytest.fixture()
def azure_openi_message_dicts_with_function_calling() -> List[dict]:
    """
    Taken from:
    - https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling.
    """
    return [
        {
            "role": "assistant",
            "function_call": {
                "name": "search_hotels",
                "arguments": '{\n  "location": "San Diego",\n  "max_price": 300,\n  "features": "beachfront,free breakfast"\n}',
            },
        }
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


def test_from_openai_message_dicts_function_calling_azure(
    azure_openi_message_dicts_with_function_calling: List[dict],
    azure_chat_messages_with_function_calling: List[ChatMessage],
) -> None:
    chat_messages = from_openai_message_dicts(
        azure_openi_message_dicts_with_function_calling
    )
    assert chat_messages == azure_chat_messages_with_function_calling


def test_create_retry_decorator() -> None:
    test_retry_decorator = create_retry_decorator(
        max_retries=6,
        random_exponential=False,
        stop_after_delay_seconds=10,
        min_seconds=2,
        max_seconds=5,
    )

    @test_retry_decorator
    def mock_function() -> str:
        # Simulate OpenAI API call with potential errors
        if mock_function.retry.statistics["attempt_number"] == 1:
            raise openai.error.Timeout(message="Timeout error")
        elif mock_function.retry.statistics["attempt_number"] == 2:
            raise openai.error.APIError(message="API error")
        elif mock_function.retry.statistics["attempt_number"] == 3:
            raise openai.error.APIConnectionError(message="API connection error")
        elif mock_function.retry.statistics["attempt_number"] == 4:
            raise openai.error.ServiceUnavailableError(
                message="Service Unavailable error"
            )
        elif mock_function.retry.statistics["attempt_number"] == 5:
            raise openai.error.RateLimitError("Rate limit error")
        else:
            # Succeed on the final attempt
            return "Success"

    # Test that the decorator retries as expected
    with pytest.raises(openai.error.RateLimitError, match="Rate limit error"):
        mock_function()
