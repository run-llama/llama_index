import inspect
import os
from collections.abc import AsyncGenerator, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.llms.vercel_ai_gateway import VercelAIGateway


@pytest.fixture()
def vercel_ai_gateway_llm():
    api_key = os.getenv("VERCEL_AI_GATEWAY_API_KEY") or os.getenv("VERCEL_OIDC_TOKEN")
    if api_key is None:
        pytest.skip(
            "VERCEL_AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN not set in environment"
        )
    return VercelAIGateway(api_key=api_key)


@pytest.fixture()
def mock_vercel_ai_gateway_llm():
    return VercelAIGateway(api_key="test")


def test_get_context_window():
    llm = VercelAIGateway(api_key="dummy", model="anthropic/claude-4-sonnet")
    assert (
        llm.context_window == 3900
    )  # Default context window from DEFAULT_CONTEXT_WINDOW
    llm.context_window = 200000
    assert llm.context_window == 200000


def test_get_all_kwargs():
    llm = VercelAIGateway(
        api_key="dummy", additional_kwargs={"foo": "bar"}, temperature=0.7
    )
    # Test that additional_kwargs are accessible
    assert llm.additional_kwargs["foo"] == "bar"
    assert llm.temperature == 0.7


def test_initialization_with_api_key():
    llm = VercelAIGateway(api_key="test-key")
    assert llm.api_key == "test-key"
    assert llm.model == "anthropic/claude-4-sonnet"
    assert llm.api_base == "https://ai-gateway.vercel.sh/v1"


def test_initialization_with_custom_model():
    llm = VercelAIGateway(api_key="test-key", model="openai/gpt-4")
    assert llm.model == "openai/gpt-4"


def test_class_name():
    llm = VercelAIGateway(api_key="test-key")
    assert llm.class_name() == "VercelAIGateway_LLM"


def test_chat(vercel_ai_gateway_llm):
    messages = [
        ChatMessage(role="system", content="Be precise and concise."),
        ChatMessage(role="user", content="Tell me 5 sentences about AI."),
    ]
    response = vercel_ai_gateway_llm.chat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip()


def test_complete(vercel_ai_gateway_llm):
    prompt = "Artificial Intelligence is a field that focuses on"
    response = vercel_ai_gateway_llm.complete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip()


def test_stream_chat(vercel_ai_gateway_llm):
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(
            role="user", content="Name the first 5 elements in the periodic table."
        ),
    ]
    stream = vercel_ai_gateway_llm.stream_chat(messages)
    assert inspect.isgenerator(stream)
    response = ""
    for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


def test_stream_complete(vercel_ai_gateway_llm):
    prompt = "List the first 5 planets in the solar system:"
    stream = vercel_ai_gateway_llm.stream_complete(prompt)
    assert inspect.isgenerator(stream)
    response = ""
    for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


@pytest.mark.asyncio
async def test_achat(vercel_ai_gateway_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="What is the largest planet in our solar system?",
        ),
    ]
    response = await vercel_ai_gateway_llm.achat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip()


@pytest.mark.asyncio
async def test_acomplete(vercel_ai_gateway_llm):
    prompt = "The largest planet in our solar system is"
    response = await vercel_ai_gateway_llm.acomplete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip()


@pytest.mark.asyncio
async def test_astream_chat(vercel_ai_gateway_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="Name the first 5 elements in the periodic table.",
        ),
    ]
    stream = await vercel_ai_gateway_llm.astream_chat(messages)
    assert isinstance(stream, AsyncIterator)
    response = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


@pytest.mark.asyncio
async def test_astream_complete(vercel_ai_gateway_llm):
    prompt = "List the first 5 elements in the periodic table:"
    stream = await vercel_ai_gateway_llm.astream_complete(prompt)
    assert isinstance(stream, AsyncIterator)
    response = ""
    async for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


def test_chat_mock(mock_vercel_ai_gateway_llm):
    # Mock the client.chat.completions.create method that OpenAI base class calls
    with patch.object(mock_vercel_ai_gateway_llm, "_get_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_response = type(
            "MockResponse",
            (),
            {
                "choices": [
                    type(
                        "MockChoice",
                        (),
                        {
                            "message": type(
                                "MockMessage",
                                (),
                                {
                                    "content": "mock response",
                                    "role": "assistant",
                                    "tool_calls": None,
                                    "function_call": None,
                                    "audio": None,
                                },
                            )(),
                            "logprobs": None,
                        },
                    )()
                ],
                "usage": type(
                    "MockUsage",
                    (),
                    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                )(),
            },
        )()
        mock_client.chat.completions.create.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hi")]
        result = mock_vercel_ai_gateway_llm.chat(messages)
        assert result.message.content == "mock response"


def test_complete_mock(mock_vercel_ai_gateway_llm):
    # Mock the client.chat.completions.create method since complete() converts to chat
    with patch.object(mock_vercel_ai_gateway_llm, "_get_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_response = type(
            "MockResponse",
            (),
            {
                "choices": [
                    type(
                        "MockChoice",
                        (),
                        {
                            "message": type(
                                "MockMessage",
                                (),
                                {
                                    "content": "mock completion",
                                    "role": "assistant",
                                    "tool_calls": None,
                                    "function_call": None,
                                    "audio": None,
                                },
                            )(),
                            "logprobs": None,
                        },
                    )()
                ],
                "usage": type(
                    "MockUsage",
                    (),
                    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                )(),
            },
        )()
        mock_client.chat.completions.create.return_value = mock_response

        result = mock_vercel_ai_gateway_llm.complete("hello")
        assert result.text == "mock completion"


@pytest.mark.asyncio
async def test_achat_mock(mock_vercel_ai_gateway_llm):
    with patch.object(mock_vercel_ai_gateway_llm, "_get_aclient") as mock_get_aclient:
        mock_client = mock_get_aclient.return_value
        mock_response = type(
            "MockResponse",
            (),
            {
                "choices": [
                    type(
                        "MockChoice",
                        (),
                        {
                            "message": type(
                                "MockMessage",
                                (),
                                {
                                    "content": "mock async",
                                    "role": "assistant",
                                    "tool_calls": None,
                                    "function_call": None,
                                    "audio": None,
                                },
                            )(),
                            "logprobs": None,
                        },
                    )()
                ],
                "usage": type(
                    "MockUsage",
                    (),
                    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                )(),
            },
        )()

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="Hi")]
        result = await mock_vercel_ai_gateway_llm.achat(messages)
        assert result.message.content == "mock async"


@pytest.mark.asyncio
async def test_acomplete_mock(mock_vercel_ai_gateway_llm):
    with patch.object(mock_vercel_ai_gateway_llm, "_get_aclient") as mock_get_aclient:
        mock_client = mock_get_aclient.return_value
        mock_response = type(
            "MockResponse",
            (),
            {
                "choices": [
                    type(
                        "MockChoice",
                        (),
                        {
                            "message": type(
                                "MockMessage",
                                (),
                                {
                                    "content": "mock async completion",
                                    "role": "assistant",
                                    "tool_calls": None,
                                    "function_call": None,
                                    "audio": None,
                                },
                            )(),
                            "logprobs": None,
                        },
                    )()
                ],
                "usage": type(
                    "MockUsage",
                    (),
                    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                )(),
            },
        )()

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await mock_vercel_ai_gateway_llm.acomplete("hello")
        assert result.text == "mock async completion"


def test_stream_chat_mock(mock_vercel_ai_gateway_llm):
    with patch.object(mock_vercel_ai_gateway_llm, "_get_client") as mock_get_client:
        mock_client = mock_get_client.return_value

        # Create mock streaming response
        def mock_stream_response():
            chunk1 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "Hello ",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            chunk2 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "world",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            yield chunk1
            yield chunk2

        mock_client.chat.completions.create.return_value = mock_stream_response()

        messages = [ChatMessage(role="user", content="Hi")]
        result = "".join(
            chunk.delta for chunk in mock_vercel_ai_gateway_llm.stream_chat(messages)
        )
        assert result == "Hello world"


def test_stream_complete_mock(mock_vercel_ai_gateway_llm):
    with patch.object(mock_vercel_ai_gateway_llm, "_get_client") as mock_get_client:
        mock_client = mock_get_client.return_value

        # Create mock streaming response
        def mock_stream_response():
            chunk1 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "Hi ",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            chunk2 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "there",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            yield chunk1
            yield chunk2

        mock_client.chat.completions.create.return_value = mock_stream_response()

        result = "".join(
            chunk.delta for chunk in mock_vercel_ai_gateway_llm.stream_complete("Yo")
        )
        assert result == "Hi there"


@pytest.mark.asyncio
async def test_astream_chat_mock(mock_vercel_ai_gateway_llm):
    with patch.object(mock_vercel_ai_gateway_llm, "_get_aclient") as mock_get_aclient:
        mock_client = mock_get_aclient.return_value

        # Create mock async streaming response
        async def mock_astream_response():
            chunk1 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "Mocked ",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            chunk2 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "streamed ",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            chunk3 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "chat",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            yield chunk1
            yield chunk2
            yield chunk3

        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_astream_response()
        )

        messages = [
            ChatMessage(role=MessageRole.USER, content="Test message 1"),
            ChatMessage(role=MessageRole.USER, content="Test message 2"),
        ]

        stream = await mock_vercel_ai_gateway_llm.astream_chat(messages)
        assert isinstance(stream, AsyncGenerator)
        full_response = ""
        async for each in stream:
            full_response += each.delta

        assert full_response == "Mocked streamed chat"


@pytest.mark.asyncio
async def test_astream_complete_mock(mock_vercel_ai_gateway_llm):
    with patch.object(mock_vercel_ai_gateway_llm, "_get_aclient") as mock_get_aclient:
        mock_client = mock_get_aclient.return_value

        # Create mock async streaming response
        async def mock_astream_response():
            chunk1 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "Mocked ",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            chunk2 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "streamed ",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            chunk3 = type(
                "MockChunk",
                (),
                {
                    "choices": [
                        type(
                            "MockChoice",
                            (),
                            {
                                "delta": type(
                                    "MockDelta",
                                    (),
                                    {
                                        "content": "completion",
                                        "tool_calls": None,
                                        "function_call": None,
                                        "role": None,
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
            yield chunk1
            yield chunk2
            yield chunk3

        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_astream_response()
        )

        prompt = "Test prompt"
        stream = await mock_vercel_ai_gateway_llm.astream_complete(prompt)
        assert isinstance(stream, AsyncGenerator)
        full_response = ""
        async for each in stream:
            full_response += each.delta
        assert full_response == "Mocked streamed completion"


def test_environment_variable_fallback():
    """Test that the LLM can be initialized using environment variables."""
    with patch.dict(os.environ, {"VERCEL_AI_GATEWAY_API_KEY": "env-key"}):
        llm = VercelAIGateway()
        assert llm.api_key == "env-key"


def test_oidc_token_fallback():
    """Test that the LLM falls back to OIDC token when API key is not available."""
    with patch.dict(os.environ, {"VERCEL_OIDC_TOKEN": "oidc-token"}, clear=True):
        llm = VercelAIGateway()
        assert llm.api_key == "oidc-token"


def test_custom_api_base():
    """Test that custom API base can be set."""
    custom_base = "https://custom.vercel.ai/v1"
    llm = VercelAIGateway(api_key="test", api_base=custom_base)
    assert llm.api_base == custom_base


def test_custom_context_window():
    """Test that custom context window can be set."""
    llm = VercelAIGateway(api_key="test", context_window=100000)
    assert llm.context_window == 100000
