import pytest
import os
import inspect
from typing import AsyncIterator, Iterator

from unittest import mock

from llama_index.llms.reka import RekaLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)


@pytest.fixture()
def reka_llm():
    api_key = os.getenv("REKA_API_KEY")
    if not api_key:
        pytest.skip("REKA_API_KEY not set in environment variables")
    return RekaLLM(model="reka-core-20240501", api_key=api_key)


# Actual integration tests


def test_chat(reka_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
    ]
    response = reka_llm.chat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip(), "Chat response should not be empty"
    print(f"\nChat response:\n{response.message.content}")


def test_complete(reka_llm):
    prompt = "The capital of France is"
    response = reka_llm.complete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip(), "Completion response should not be empty"
    print(f"\nCompletion response:\n{response.text}")


def test_stream_chat(reka_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="List the first 5 planets in the solar system.",
        ),
    ]
    stream = reka_llm.stream_chat(messages)
    assert inspect.isgenerator(stream), "stream_chat should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Streamed chat response should not be empty"
    print(f"\n\nFull streamed chat response:\n{full_response}")


def test_stream_complete(reka_llm):
    prompt = "List the first 5 planets in the solar system:"
    stream = reka_llm.stream_complete(prompt)
    assert inspect.isgenerator(stream), "stream_complete should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Streamed completion response should not be empty"
    print(f"\n\nFull streamed completion response:\n{full_response}")


@pytest.mark.asyncio
async def test_achat(reka_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="What is the largest planet in our solar system?",
        ),
    ]
    response = await reka_llm.achat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip(), "Async chat response should not be empty"
    print(f"\nAsync chat response:\n{response.message.content}")


@pytest.mark.asyncio
async def test_acomplete(reka_llm):
    prompt = "The largest planet in our solar system is"
    response = await reka_llm.acomplete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip(), "Async completion response should not be empty"
    print(f"\nAsync completion response:\n{response.text}")


@pytest.mark.asyncio
async def test_astream_chat(reka_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="Name the first 5 elements in the periodic table.",
        ),
    ]
    stream = await reka_llm.astream_chat(messages)
    assert isinstance(stream, AsyncIterator), (
        "astream_chat should return an async generator"
    )

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Async streamed chat response should not be empty"
    print(f"\n\nFull async streamed chat response:\n{full_response}")


@pytest.mark.asyncio
async def test_astream_complete(reka_llm):
    prompt = "List the first 5 elements in the periodic table:"
    stream = await reka_llm.astream_complete(prompt)
    assert isinstance(stream, AsyncIterator), (
        "astream_complete should return an async generator"
    )

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), (
        "Async streamed completion response should not be empty"
    )
    print(f"\n\nFull async streamed completion response:\n{full_response}")


class MockRekaStreamResponse:
    def __init__(self, content: str):
        self.chunks = [
            mock.MagicMock(
                responses=[mock.MagicMock(chunk=mock.MagicMock(content=token))]
            )
            for token in content.split()
        ]

    def __iter__(self) -> Iterator[mock.MagicMock]:
        return iter(self.chunks)

    async def __aiter__(self) -> AsyncIterator[mock.MagicMock]:
        for chunk in self.chunks:
            yield chunk


@pytest.fixture()
def mock_reka_llm():
    return RekaLLM(api_key="dummy", temperature=0.3)


def test_chat_mock(mock_reka_llm):
    with mock.patch.object(mock_reka_llm._client.chat, "create") as mock_create:
        mock_create.return_value = mock.MagicMock(
            responses=[
                mock.MagicMock(message=mock.MagicMock(content="Mocked chat response"))
            ]
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
        response = mock_reka_llm.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.message.content == "Mocked chat response"
        assert response.message.role == MessageRole.ASSISTANT
        mock_create.assert_called_once()


def test_complete_mock(mock_reka_llm):
    with mock.patch.object(mock_reka_llm._client.chat, "create") as mock_create:
        mock_create.return_value = mock.MagicMock(
            responses=[
                mock.MagicMock(
                    message=mock.MagicMock(content="Mocked completion response")
                )
            ]
        )

        response = mock_reka_llm.complete("Test prompt")

        assert isinstance(response, CompletionResponse)
        assert response.text == "Mocked completion response"
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_achat_mock(mock_reka_llm):
    with mock.patch.object(mock_reka_llm._aclient.chat, "create") as mock_acreate:
        mock_acreate.return_value = mock.MagicMock(
            responses=[
                mock.MagicMock(
                    message=mock.MagicMock(content="Mocked async chat response")
                )
            ]
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Async test")]
        response = await mock_reka_llm.achat(messages)

        assert isinstance(response, ChatResponse)
        assert response.message.content == "Mocked async chat response"
        assert response.message.role == MessageRole.ASSISTANT
        mock_acreate.assert_called_once()


@pytest.mark.asyncio
async def test_acomplete_mock(mock_reka_llm):
    with mock.patch.object(mock_reka_llm._aclient.chat, "create") as mock_acreate:
        mock_acreate.return_value = mock.MagicMock(
            responses=[
                mock.MagicMock(
                    message=mock.MagicMock(content="Mocked async completion response")
                )
            ]
        )

        response = await mock_reka_llm.acomplete("Async test prompt")

        assert isinstance(response, CompletionResponse)
        assert response.text == "Mocked async completion response"
        mock_acreate.assert_called_once()


def test_stream_chat_mock(mock_reka_llm):
    with mock.patch.object(
        mock_reka_llm._client.chat, "create_stream"
    ) as mock_create_stream:
        mock_create_stream.return_value = MockRekaStreamResponse(
            "Mocked streaming response"
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Stream test")]
        stream = mock_reka_llm.stream_chat(messages)

        response = list(stream)
        assert all(isinstance(chunk, ChatResponse) for chunk in response)
        assert [chunk.message.content for chunk in response] == [
            "Mocked",
            "streaming",
            "response",
        ]
        mock_create_stream.assert_called_once()


def test_stream_complete_mock(mock_reka_llm):
    with mock.patch.object(
        mock_reka_llm._client.chat, "create_stream"
    ) as mock_create_stream:
        mock_create_stream.return_value = MockRekaStreamResponse(
            "Mocked streaming completion"
        )

        stream = mock_reka_llm.stream_complete("Test prompt")

        response = list(stream)
        assert all(isinstance(chunk, CompletionResponse) for chunk in response)
        assert [chunk.text for chunk in response] == [
            "Mocked",
            "streaming",
            "completion",
        ]
        mock_create_stream.assert_called_once()


@pytest.mark.asyncio
async def test_astream_chat_mock(mock_reka_llm):
    with mock.patch.object(
        mock_reka_llm._aclient.chat, "create_stream"
    ) as mock_acreate_stream:
        mock_acreate_stream.return_value = MockRekaStreamResponse(
            "Mocked async streaming response"
        )

        messages = [ChatMessage(role=MessageRole.USER, content="Async stream test")]
        stream = await mock_reka_llm.astream_chat(messages)

        response = [chunk async for chunk in stream]
        assert all(isinstance(chunk, ChatResponse) for chunk in response)
        assert [chunk.message.content for chunk in response] == [
            "Mocked",
            "async",
            "streaming",
            "response",
        ]
        mock_acreate_stream.assert_called_once()


@pytest.mark.asyncio
async def test_astream_complete_mock(mock_reka_llm):
    with mock.patch.object(
        mock_reka_llm._aclient.chat, "create_stream"
    ) as mock_acreate_stream:
        mock_acreate_stream.return_value = MockRekaStreamResponse(
            "Mocked async streaming response"
        )

        stream = await mock_reka_llm.astream_complete("Async stream test prompt")

        response = [chunk async for chunk in stream]
        assert all(isinstance(chunk, CompletionResponse) for chunk in response)
        assert [chunk.text for chunk in response] == [
            "Mocked",
            "async",
            "streaming",
            "response",
        ]
        mock_acreate_stream.assert_called_once()
