import inspect
from typing import AsyncIterator
import pytest
import os

from unittest import mock
from unittest.mock import AsyncMock, patch, Mock

from llama_index.llms.perplexity import Perplexity
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)


@pytest.fixture()
def perplexity_llm():
    api_key = os.getenv("PPLX_API_KEY")
    if api_key is None:
        pytest.skip("PPLX_API_KEY not set in environment")
    return Perplexity(api_key=api_key)


def test_chat(perplexity_llm):
    messages_dict = [
        {"role": "system", "content": "Be precise and concise."},
        {"role": "user", "content": "Tell me 5 sentences about Perplexity."},
    ]
    messages = [ChatMessage(**msg) for msg in messages_dict]
    response = perplexity_llm.chat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip(), "Chat response should not be empty"
    print(f"\nChat response:\n{response.message.content}")


def test_complete(perplexity_llm):
    prompt = "Perplexity is a company that provides"
    response = perplexity_llm.complete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip(), "Completion response should not be empty"
    print(f"\nCompletion response:\n{response.text}")


def test_stream_chat(perplexity_llm):
    messages_dict = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Name the first 5 elements in the periodic table."},
    ]
    messages = [ChatMessage(**msg) for msg in messages_dict]
    stream = perplexity_llm.stream_chat(messages)
    assert inspect.isgenerator(stream), "stream_chat should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Streamed chat response should not be empty"
    print(f"\n\nFull streamed chat response:\n{full_response}")


def test_stream_complete(perplexity_llm):
    prompt = "List the first 5 planets in the solar system:"
    stream = perplexity_llm.stream_complete(prompt)
    assert inspect.isgenerator(stream), "stream_complete should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Streamed completion response should not be empty"
    print(f"\n\nFull streamed completion response:\n{full_response}")


@pytest.mark.asyncio()
async def test_achat(perplexity_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="What is the largest planet in our solar system?",
        ),
    ]
    response = await perplexity_llm.achat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip(), "Async chat response should not be empty"
    print(f"\nAsync chat response:\n{response.message.content}")


@pytest.mark.asyncio()
async def test_acomplete(perplexity_llm):
    prompt = "The largest planet in our solar system is"
    response = await perplexity_llm.acomplete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip(), "Async completion response should not be empty"
    print(f"\nAsync completion response:\n{response.text}")


@pytest.mark.asyncio()
async def test_astream_chat(perplexity_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="Name the first 5 elements in the periodic table.",
        ),
    ]
    stream = await perplexity_llm.astream_chat(messages)
    assert isinstance(
        stream, AsyncIterator
    ), "astream_chat should return an async generator"

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Async streamed chat response should not be empty"
    print(f"\n\nFull async streamed chat response:\n{full_response}")


@pytest.mark.asyncio()
async def test_astream_complete(perplexity_llm):
    prompt = "List the first 5 elements in the periodic table:"
    stream = await perplexity_llm.astream_complete(prompt)
    assert isinstance(
        stream, AsyncIterator
    ), "astream_complete should return an async generator"

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert (
        full_response.strip()
    ), "Async streamed completion response should not be empty"
    print(f"\n\nFull async streamed completion response:\n{full_response}")


@pytest.fixture()
def mock_perplexity_llm():
    return Perplexity(api_key="dummy", temperature=0.3)


def test_chat_mock(mock_perplexity_llm):
    with mock.patch.object(mock_perplexity_llm, "_chat") as mock_chat:
        mock_chat.return_value = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content="Mocked chat response"
            ),
            raw={},
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
        response = mock_perplexity_llm.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.message.content == "Mocked chat response"
        assert response.message.role == MessageRole.ASSISTANT
        mock_chat.assert_called_once()


def test_complete_mock(mock_perplexity_llm):
    with mock.patch.object(mock_perplexity_llm, "_complete") as mock_complete:
        mock_complete.return_value = CompletionResponse(
            text="Mocked completion response", raw={}
        )

        response = mock_perplexity_llm.complete("Test prompt")

        assert isinstance(response, CompletionResponse)
        assert response.text == "Mocked completion response"
        mock_complete.assert_called_once()


@pytest.mark.asyncio()
async def test_achat_mock(mock_perplexity_llm):
    with mock.patch.object(mock_perplexity_llm, "_achat") as mock_achat:
        mock_achat.return_value = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content="Mocked async chat response"
            ),
            raw={},
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Async test")]
        response = await mock_perplexity_llm.achat(messages)

        assert isinstance(response, ChatResponse)
        assert response.message.content == "Mocked async chat response"
        assert response.message.role == MessageRole.ASSISTANT
        mock_achat.assert_called_once()


@pytest.mark.asyncio()
async def test_acomplete_mock(mock_perplexity_llm):
    with mock.patch.object(mock_perplexity_llm, "_acomplete") as mock_aacomplete:
        mock_aacomplete.return_value = CompletionResponse(
            text="Mocked async completion response", raw={}
        )

        response = await mock_perplexity_llm.acomplete("Async test prompt")

        assert isinstance(response, CompletionResponse)
        assert response.text == "Mocked async completion response"
        mock_aacomplete.assert_called_once()


def test_stream_complete_mock(mock_perplexity_llm):
    with mock.patch.object(
        mock_perplexity_llm, "_stream_complete"
    ) as mock_stream_complete:
        mock_stream_complete.return_value = [
            CompletionResponse(delta="Mocked ", text="Mocked "),
            CompletionResponse(delta="streamed ", text="Mocked streamed "),
            CompletionResponse(delta="completion", text="Mocked streamed completion"),
        ]

        stream = mock_perplexity_llm._stream_complete("Test prompt")
        full_response = stream[-1].text

        assert full_response == "Mocked streamed completion"
        mock_stream_complete.assert_called_once()


@pytest.mark.asyncio()
async def test_astream_complete(mock_perplexity_llm):
    prompt = "Test prompt"
    mock_astream_complete = AsyncMock(
        return_value=[
            CompletionResponse(delta="Mocked ", text="Mocked "),
            CompletionResponse(delta="streamed ", text="Mocked streamed "),
            CompletionResponse(delta="completion", text="Mocked streamed completion"),
        ]
    )

    with patch.object(mock_perplexity_llm, "_astream_complete", mock_astream_complete):
        stream = await mock_perplexity_llm._astream_complete(prompt)
        full_response = stream[-1].text

        assert full_response == "Mocked streamed completion"
        mock_astream_complete.assert_called_once_with(prompt)


def test_stream_chat(mock_perplexity_llm):
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

    with patch.object(mock_perplexity_llm, "_stream_chat", mock_stream_chat):
        stream = mock_perplexity_llm._stream_chat(messages)
        # full_response = "".join(chunk.message.content for chunk in stream)
        full_response = stream[-1].message.content

        assert full_response == "Mocked streamed chat"
        mock_stream_chat.assert_called_once_with(messages)


@pytest.mark.asyncio()
async def test_astream_chat(mock_perplexity_llm):
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

    with patch.object(mock_perplexity_llm, "_astream_chat", mock_astream_chat):
        stream = await mock_perplexity_llm._astream_chat(messages)
        full_response = stream[-1].message.content

        assert full_response == "Mocked streamed chat"
        mock_astream_chat.assert_called_once_with(messages)
