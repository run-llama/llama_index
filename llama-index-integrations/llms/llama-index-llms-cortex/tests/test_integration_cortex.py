import inspect
import os
from typing import AsyncIterator
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.llms.cortex import Cortex


@pytest.fixture()
def cortex_llm() -> Cortex:
    user = os.getenv("SNOWFLAKE_USERNAME")
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    key_file = os.getenv("SNOWFLAKE_KEY_FILE")
    if user is None or account is None or key_file is None:
        pytest.skip("Environment variables not set.")
    return Cortex(user=user, account=account, private_key_file=key_file)


def test_complete(cortex_llm):
    response = cortex_llm.complete("hello", temperature=0, max_tokens=2)
    assert isinstance(response, CompletionResponse)
    assert "hello" in response.text.lower()


@pytest.mark.asyncio()
async def test_acomplete(cortex_llm):
    response = await cortex_llm.acomplete("hello")
    assert isinstance(response, CompletionResponse)
    assert "hello" in response.text.lower()


def test_stream_complete(cortex_llm):
    stream = cortex_llm.stream_complete("hello", temperature=0, max_tokens=2)
    assert inspect.isgenerator(stream), "stream_complete should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip()


@pytest.mark.asyncio()
async def test_astream_complete(cortex_llm):
    stream = await cortex_llm.astream_complete("hello")
    assert isinstance(stream, AsyncIterator)

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip()


def test_chat(cortex_llm):
    messages_dict = [
        {"role": "system", "content": "You are a poet."},
        {"role": "user", "content": "Write me a haiky about snowflakes."},
    ]
    messages = [ChatMessage(**msg) for msg in messages_dict]
    response = cortex_llm.chat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip()


@pytest.mark.asyncio()
async def test_achat(cortex_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="What is the largest planet in our solar system?",
        ),
    ]
    response = await cortex_llm.achat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip()


def test_stream_chat(cortex_llm):
    messages_dict = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Name the first 5 elements in the periodic table."},
    ]
    messages = [ChatMessage(**msg) for msg in messages_dict]
    stream = cortex_llm.stream_chat(messages)
    assert inspect.isgenerator(stream)

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip()


@pytest.mark.asyncio()
async def test_astream_chat(cortex_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="Name the first 5 elements in the periodic table.",
        ),
    ]
    stream = await cortex_llm.astream_chat(messages)
    assert isinstance(stream, AsyncIterator)

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip()


@pytest.fixture()
def mock_cortex_llm() -> Cortex:
    return Cortex(
        user="example",
        account="my_sf_account",
        private_key_file="/dummy/path/to/key.p8",
    )


def test_complete_mock(mock_cortex_llm):
    with mock.patch.object(mock_cortex_llm, "_complete") as mock_complete:
        mock_complete.return_value = CompletionResponse(
            text="Mocked completion response", raw={}
        )

        response = mock_cortex_llm.complete("Test prompt")

        assert isinstance(response, CompletionResponse)
        assert response.text == "Mocked completion response"
        mock_complete.assert_called_once()


@pytest.mark.asyncio()
async def test_acomplete_mock(mock_cortex_llm):
    with mock.patch.object(mock_cortex_llm, "_acomplete") as mock_acomplete:
        mock_acomplete.return_value = CompletionResponse(
            text="Mocked async completion response", raw={}
        )

        response = await mock_cortex_llm.acomplete("Async test prompt", formatted=True)

        assert isinstance(response, CompletionResponse)
        assert response.text == "Mocked async completion response"
        mock_acomplete.assert_called_once()


def test_stream_complete_mock(mock_cortex_llm):
    with mock.patch.object(mock_cortex_llm, "_stream_complete") as mock_stream_complete:
        mock_stream_complete.return_value = [
            CompletionResponse(delta="Mocked ", text="Mocked "),
            CompletionResponse(delta="streamed ", text="Mocked streamed "),
            CompletionResponse(delta="completion", text="Mocked streamed completion"),
        ]

        stream = mock_cortex_llm._stream_complete("Test prompt")
        full_response = stream[-1].text

        assert full_response == "Mocked streamed completion"
        mock_stream_complete.assert_called_once()


@pytest.mark.asyncio()
async def test_astream_complete_mock(mock_cortex_llm):
    prompt = "Test prompt"
    mock_astream_complete = AsyncMock(
        return_value=[
            CompletionResponse(delta="Mocked ", text="Mocked "),
            CompletionResponse(delta="streamed ", text="Mocked streamed "),
            CompletionResponse(delta="completion", text="Mocked streamed completion"),
        ]
    )

    with patch.object(mock_cortex_llm, "_astream_complete", mock_astream_complete):
        stream = await mock_cortex_llm._astream_complete(prompt)
        full_response = stream[-1].text

        assert full_response == "Mocked streamed completion"
        mock_astream_complete.assert_called_once_with(prompt)


def test_chat_mock(mock_cortex_llm):
    with mock.patch.object(mock_cortex_llm, "_chat") as mock_chat:
        mock_chat.return_value = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content="Mocked chat response"
            ),
            raw={},
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
        response = mock_cortex_llm.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.message.content == "Mocked chat response"
        assert response.message.role == MessageRole.ASSISTANT
        mock_chat.assert_called_once()


@pytest.mark.asyncio()
async def test_achat_mock(mock_cortex_llm):
    with mock.patch.object(mock_cortex_llm, "_achat") as mock_achat:
        mock_achat.return_value = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content="Mocked async chat response"
            ),
            raw={},
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Async test")]
        response = await mock_cortex_llm.achat(messages)

        assert isinstance(response, ChatResponse)
        assert response.message.content == "Mocked async chat response"
        assert response.message.role == MessageRole.ASSISTANT
        mock_achat.assert_called_once()


def test_stream_chat_mock(mock_cortex_llm):
    messages = [
        ChatMessage(role=MessageRole.USER, content="Test message 1"),
        ChatMessage(role=MessageRole.USER, content="Test message 2"),
    ]
    mock_stream_chat = Mock(
        return_value=[
            ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content="Mocked ", raw={}
                ),
                delta="Mocked ",
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content="Mocked streamed ", raw={}
                ),
                delta="streamed ",
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content="Mocked streamed chat", raw={}
                ),
                delta="chat",
                raw={},
            ),
        ]
    )

    with patch.object(mock_cortex_llm, "_stream_chat", mock_stream_chat):
        stream = mock_cortex_llm._stream_chat(messages)
        # full_response = "".join(chunk.message.content for chunk in stream)
        full_response = stream[-1].message.content

        assert full_response == "Mocked streamed chat"
        mock_stream_chat.assert_called_once_with(messages)


@pytest.mark.asyncio()
async def test_astream_chat_mock(mock_cortex_llm):
    messages = [
        ChatMessage(role=MessageRole.USER, content="Test message 1"),
        ChatMessage(role=MessageRole.USER, content="Test message 2"),
    ]
    mock_astream_chat = AsyncMock(
        return_value=[
            ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content="Mocked ", raw={}
                ),
                delta="Mocked ",
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content="Mocked streamed ", raw={}
                ),
                delta="streamed ",
                raw={},
            ),
            ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content="Mocked streamed chat", raw={}
                ),
                delta="chat",
                raw={},
            ),
        ]
    )

    with patch.object(mock_cortex_llm, "_astream_chat", mock_astream_chat):
        stream = await mock_cortex_llm._astream_chat(messages)
        full_response = stream[-1].message.content

        assert full_response == "Mocked streamed chat"
        mock_astream_chat.assert_called_once_with(messages)
