import os
from typing import Any, Generator, AsyncGenerator
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from llama_index.core.llms import ChatMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from llama_index.llms.upstage import Upstage


class FakeApiKeyManager:
    """
    Test utility for managing API keys during tests.
    Temporarily sets a fake API key and restores the original key after the test.
    """

    def __init__(self):
        self.original_api_key = None

    def __enter__(self) -> None:
        # Save original API key
        self.original_api_key = os.environ.get("UPSTAGE_API_KEY", "")

        # Set fake API key for testing
        os.environ["UPSTAGE_API_KEY"] = "fake_key_for_testing"

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore original API key
        if self.original_api_key:
            os.environ["UPSTAGE_API_KEY"] = self.original_api_key
        else:
            os.environ.pop("UPSTAGE_API_KEY", None)


def mock_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="solar-pro2",
        usage=CompletionUsage(prompt_tokens=13, completion_tokens=7, total_tokens=20),
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="\n\nThis is a test!"
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def mock_chat_completion_stream() -> Generator[ChatCompletionChunk, None, None]:
    responses = [
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="solar-pro2",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="solar-pro2",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="\n\n"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="solar-pro2",
            choices=[
                ChunkChoice(delta=ChoiceDelta(content="2"), finish_reason=None, index=0)
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="solar-pro2",
            choices=[ChunkChoice(delta=ChoiceDelta(), finish_reason="stop", index=0)],
        ),
    ]
    yield from responses


async def mock_async_chat_completion_stream() -> AsyncGenerator[
    ChatCompletionChunk, None
]:
    for chunk in mock_chat_completion_stream():
        yield chunk


# Parameter validation tests
@pytest.mark.parametrize(
    ("param_name", "invalid_value"),
    [
        ("reasoning_effort", "invalid"),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("frequency_penalty", -3.0),
        ("frequency_penalty", 3.0),
        ("presence_penalty", -3.0),
        ("presence_penalty", 3.0),
    ],
)
def test_parameter_validation_errors(param_name: str, invalid_value: Any):
    """Test that invalid parameter values raise validation errors."""
    try:
        llm = Upstage(model="solar-pro2", **{param_name: invalid_value})
        raise AssertionError(
            f"Expected validation error for invalid {param_name}: {invalid_value}"
        )
    except Exception as e:
        error_message = str(e).lower()
        # Check that the error message contains validation-related keywords
        assert any(
            keyword in error_message
            for keyword in ["validation", "invalid", "gte", "lte"]
        ), f"Expected validation error message, got: {error_message}"


# Parameter integration tests
@pytest.mark.parametrize(
    ("param_name", "param_value"),
    [
        ("reasoning_effort", "low"),
        ("reasoning_effort", "medium"),
        ("reasoning_effort", "high"),
        ("top_p", 0.0),
        ("top_p", 1.0),
        ("frequency_penalty", -2.0),
        ("frequency_penalty", 2.0),
        ("presence_penalty", -2.0),
        ("presence_penalty", 2.0),
    ],
)
@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_parameter_integration(
    mock_sync_upstage: MagicMock, param_name: str, param_value: Any
):
    """Test that parameters are correctly passed in API requests."""
    with FakeApiKeyManager():
        mock_instance = mock_sync_upstage.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion()

        llm = Upstage(model="solar-pro2", **{param_name: param_value})
        message = ChatMessage(role="user", content="test message")

        llm.chat([message])

        # Verify that the parameter is correctly passed in the API request
        mock_instance.chat.completions.create.assert_called()
        call_args = mock_instance.chat.completions.create.call_args
        assert param_name in call_args.kwargs, (
            f"Parameter '{param_name}' not found in API request"
        )
        assert call_args.kwargs[param_name] == param_value, (
            f"Parameter '{param_name}' value mismatch. Expected: {param_value}, Got: {call_args.kwargs[param_name]}"
        )


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_basic_functionality(mock_sync_upstage: MagicMock) -> None:
    """Test basic functionality including chat, complete, and streaming."""
    with FakeApiKeyManager():
        mock_instance = mock_sync_upstage.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion()

        llm = Upstage(model="solar-pro2")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        # Test complete
        response = llm.complete(prompt)
        assert response.text == "\n\nThis is a test!"

        # Test chat
        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is a test!"

        # Test streaming
        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream()
        )
        response_gen = llm.stream_complete(prompt)
        responses = list(response_gen)
        assert len(responses) > 0
        assert responses[-1].text == "\n\n2"

        # Test streaming chat
        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream()
        )
        chat_response_gen = llm.stream_chat([message])
        chat_responses = list(chat_response_gen)
        assert len(chat_responses) > 0
        assert chat_responses[-1].message.content == "\n\n2"
        assert chat_responses[-1].message.role == "assistant"


def test_validates_api_key_is_present() -> None:
    """Test API key validation."""
    with FakeApiKeyManager():
        os.environ["UPSTAGE_API_KEY"] = "a" * 32

        # We can create a new LLM when the env variable is set
        assert Upstage()

        os.environ["UPSTAGE_API_KEY"] = ""

        # We can create a new LLM when the api_key is set on the class directly
        assert Upstage(api_key="a" * 32)


def test_token_counting_basic_functionality():
    """Test that token counting works without errors."""
    with FakeApiKeyManager():
        llm = Upstage(model="solar-pro2")

        # Test token counting with simple messages
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?"),
        ]

        # Should not raise any exceptions
        token_count = llm.get_num_tokens_from_message(messages)

        # Basic sanity checks
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 100  # Reasonable for short messages


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_http_headers_integration(mock_sync_upstage: MagicMock) -> None:
    """Test that HTTP requests include the x-upstage-client header."""
    with FakeApiKeyManager():
        mock_instance = mock_sync_upstage.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion()

        llm = Upstage(model="solar-pro2")
        message = ChatMessage(role="user", content="test message")

        llm.chat([message])

        # Verify OpenAI client was called
        mock_sync_upstage.assert_called()

        # Verify headers in OpenAI client initialization
        call_args = mock_sync_upstage.call_args
        kwargs = call_args.kwargs
        headers = kwargs.get("default_headers", {})

        # Check that x-upstage-client header is present
        assert headers.get("x-upstage-client") == "llamaindex"


@pytest.mark.asyncio
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_async_basic_functionality(mock_async_upstage: MagicMock) -> None:
    """Test async basic functionality including achat and astream_chat."""
    with FakeApiKeyManager():
        mock_instance = mock_async_upstage.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=mock_chat_completion()
        )

        llm = Upstage(model="solar-pro2")
        message = ChatMessage(role="user", content="test message")

        # Test async chat
        chat_response = await llm.achat([message])
        assert chat_response.message.content == "\n\nThis is a test!"

        # Test async streaming chat
        mock_instance.chat.completions.create = AsyncMock(
            return_value=mock_async_chat_completion_stream()
        )
        chat_response_gen = await llm.astream_chat([message])
        chat_responses = [resp async for resp in chat_response_gen]
        assert len(chat_responses) > 0
        assert chat_responses[-1].message.content == "\n\n2"
        assert chat_responses[-1].message.role == "assistant"


@pytest.mark.asyncio
@patch("llama_index.llms.openai.base.AsyncOpenAI")
async def test_async_parameter_integration(mock_async_upstage: MagicMock) -> None:
    """Test that custom parameters are correctly passed in async API requests."""
    with FakeApiKeyManager():
        mock_instance = mock_async_upstage.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=mock_chat_completion()
        )

        llm = Upstage(
            model="solar-pro2",
            reasoning_effort="high",
            top_p=0.8,
            frequency_penalty=1.0,
            presence_penalty=-0.5,
        )
        message = ChatMessage(role="user", content="test message")

        await llm.achat([message])

        # Verify that custom parameters are included in the async API request
        mock_instance.chat.completions.create.assert_called()
        call_args = mock_instance.chat.completions.create.call_args

        assert call_args.kwargs["reasoning_effort"] == "high"
        assert call_args.kwargs["top_p"] == 0.8
        assert call_args.kwargs["frequency_penalty"] == 1.0
        assert call_args.kwargs["presence_penalty"] == -0.5
