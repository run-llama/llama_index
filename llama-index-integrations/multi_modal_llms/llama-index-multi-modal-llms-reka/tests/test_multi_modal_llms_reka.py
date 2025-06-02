import pytest
import os
import inspect
from typing import AsyncIterator, Iterator

from unittest import mock

from llama_index.multi_modal_llms.reka import RekaMultiModalLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.schema import ImageDocument


@pytest.fixture()
def reka_mm_llm():
    api_key = os.getenv("REKA_API_KEY")
    if not api_key:
        pytest.skip("REKA_API_KEY not set in environment variables")
    return RekaMultiModalLLM(model="reka-core-20240501", api_key=api_key)


@pytest.fixture()
def image_document():
    return ImageDocument(image_url="https://v0.docs.reka.ai/_images/000000245576.jpg")


def test_chat_with_image(reka_mm_llm, image_document):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What do you see in this image?"),
    ]
    response = reka_mm_llm.chat(messages, image_documents=[image_document])
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip(), "Chat response should not be empty"
    print(f"\nChat response with image:\n{response.message.content}")


def test_complete_with_image(reka_mm_llm, image_document):
    prompt = "Describe the animal in this image:"
    response = reka_mm_llm.complete(prompt, image_documents=[image_document])
    assert isinstance(response, CompletionResponse)
    assert response.text.strip(), "Completion response should not be empty"
    print(f"\nCompletion response with image:\n{response.text}")


def test_stream_chat_with_image(reka_mm_llm, image_document):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER, content="What is the animal doing in this image?"
        ),
    ]
    stream = reka_mm_llm.stream_chat(messages, image_documents=[image_document])
    assert inspect.isgenerator(stream), "stream_chat should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Streamed chat response should not be empty"
    print(f"\n\nFull streamed chat response with image:\n{full_response}")


def test_stream_complete_with_image(reka_mm_llm, image_document):
    prompt = "Describe the colors in this image:"
    stream = reka_mm_llm.stream_complete(prompt, image_documents=[image_document])
    assert inspect.isgenerator(stream), "stream_complete should return a generator"

    full_response = ""
    for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Streamed completion response should not be empty"
    print(f"\n\nFull streamed completion response with image:\n{full_response}")


@pytest.mark.asyncio
async def test_achat_with_image(reka_mm_llm, image_document):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER, content="What breed of cat is in this image?"
        ),
    ]
    response = await reka_mm_llm.achat(messages, image_documents=[image_document])
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip(), "Async chat response should not be empty"
    print(f"\nAsync chat response with image:\n{response.message.content}")


@pytest.mark.asyncio
async def test_acomplete_with_image(reka_mm_llm, image_document):
    prompt = "Describe the background of this image:"
    response = await reka_mm_llm.acomplete(prompt, image_documents=[image_document])
    assert isinstance(response, CompletionResponse)
    assert response.text.strip(), "Async completion response should not be empty"
    print(f"\nAsync completion response with image:\n{response.text}")


@pytest.mark.asyncio
async def test_astream_chat_with_image(reka_mm_llm, image_document):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER, content="What objects are visible in this image?"
        ),
    ]
    stream = await reka_mm_llm.astream_chat(messages, image_documents=[image_document])
    assert isinstance(stream, AsyncIterator), (
        "astream_chat should return an async generator"
    )

    full_response = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        full_response += chunk.delta
        print(chunk.delta, end="", flush=True)

    assert full_response.strip(), "Async streamed chat response should not be empty"
    print(f"\n\nFull async streamed chat response with image:\n{full_response}")


@pytest.mark.asyncio
async def test_astream_complete_with_image(reka_mm_llm, image_document):
    prompt = "List the colors present in this image:"
    stream = await reka_mm_llm.astream_complete(
        prompt, image_documents=[image_document]
    )
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
    print(f"\n\nFull async streamed completion response with image:\n{full_response}")


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
def mock_reka_mm_llm():
    return RekaMultiModalLLM(api_key="dummy", temperature=0.3)


@pytest.fixture()
def mock_image_document():
    return ImageDocument(image_url="https://v0.docs.reka.ai/_images/000000245576.jpg")


def test_chat_mock_with_image(mock_reka_mm_llm, mock_image_document):
    with mock.patch.object(mock_reka_mm_llm._client.chat, "create") as mock_create:
        mock_create.return_value = mock.MagicMock(
            responses=[
                mock.MagicMock(
                    message=mock.MagicMock(
                        content="Mocked chat response about an image of a cat"
                    )
                )
            ]
        )

        messages = [ChatMessage(role=MessageRole.USER, content="What's in this image?")]
        response = mock_reka_mm_llm.chat(
            messages, image_documents=[mock_image_document]
        )

        assert isinstance(response, ChatResponse)
        assert (
            response.message.content == "Mocked chat response about an image of a cat"
        )
        assert response.message.role == MessageRole.ASSISTANT
        mock_create.assert_called_once()


def test_complete_mock_with_image(mock_reka_mm_llm, mock_image_document):
    with mock.patch.object(mock_reka_mm_llm._client.chat, "create") as mock_create:
        mock_create.return_value = mock.MagicMock(
            responses=[
                mock.MagicMock(
                    message=mock.MagicMock(
                        content="Mocked completion response describing a cat image"
                    )
                )
            ]
        )

        response = mock_reka_mm_llm.complete(
            "Describe this image:", image_documents=[mock_image_document]
        )

        assert isinstance(response, CompletionResponse)
        assert response.text == "Mocked completion response describing a cat image"
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_achat_mock_with_image(mock_reka_mm_llm, mock_image_document):
    with mock.patch.object(mock_reka_mm_llm._aclient.chat, "create") as mock_acreate:
        mock_acreate.return_value = mock.MagicMock(
            responses=[
                mock.MagicMock(
                    message=mock.MagicMock(
                        content="Mocked async chat response about a cat image"
                    )
                )
            ]
        )

        messages = [ChatMessage(role=MessageRole.USER, content="What's in this image?")]
        response = await mock_reka_mm_llm.achat(
            messages, image_documents=[mock_image_document]
        )

        assert isinstance(response, ChatResponse)
        assert (
            response.message.content == "Mocked async chat response about a cat image"
        )
        assert response.message.role == MessageRole.ASSISTANT
        mock_acreate.assert_called_once()


@pytest.mark.asyncio
async def test_acomplete_mock_with_image(mock_reka_mm_llm, mock_image_document):
    with mock.patch.object(mock_reka_mm_llm._aclient.chat, "create") as mock_acreate:
        mock_acreate.return_value = mock.MagicMock(
            responses=[
                mock.MagicMock(
                    message=mock.MagicMock(
                        content="Mocked async completion response about a cat image"
                    )
                )
            ]
        )

        response = await mock_reka_mm_llm.acomplete(
            "Describe this image:", image_documents=[mock_image_document]
        )

        assert isinstance(response, CompletionResponse)
        assert response.text == "Mocked async completion response about a cat image"
        mock_acreate.assert_called_once()


def test_stream_chat_mock_with_image(mock_reka_mm_llm, mock_image_document):
    with mock.patch.object(
        mock_reka_mm_llm._client.chat, "create_stream"
    ) as mock_create_stream:
        mock_create_stream.return_value = MockRekaStreamResponse(
            "Mocked streaming response about a cat image"
        )

        messages = [ChatMessage(role=MessageRole.USER, content="What's in this image?")]
        stream = mock_reka_mm_llm.stream_chat(
            messages, image_documents=[mock_image_document]
        )

        response = list(stream)
        assert all(isinstance(chunk, ChatResponse) for chunk in response)
        assert [chunk.message.content for chunk in response] == [
            "Mocked",
            "streaming",
            "response",
            "about",
            "a",
            "cat",
            "image",
        ]
        mock_create_stream.assert_called_once()


def test_stream_complete_mock_with_image(mock_reka_mm_llm, mock_image_document):
    with mock.patch.object(
        mock_reka_mm_llm._client.chat, "create_stream"
    ) as mock_create_stream:
        mock_create_stream.return_value = MockRekaStreamResponse(
            "Mocked streaming completion about a cat image"
        )

        stream = mock_reka_mm_llm.stream_complete(
            "Describe this image:", image_documents=[mock_image_document]
        )

        response = list(stream)
        assert all(isinstance(chunk, CompletionResponse) for chunk in response)
        assert [chunk.text for chunk in response] == [
            "Mocked",
            "streaming",
            "completion",
            "about",
            "a",
            "cat",
            "image",
        ]
        mock_create_stream.assert_called_once()


@pytest.mark.asyncio
async def test_astream_chat_mock_with_image(mock_reka_mm_llm, mock_image_document):
    with mock.patch.object(
        mock_reka_mm_llm._aclient.chat, "create_stream"
    ) as mock_acreate_stream:
        mock_acreate_stream.return_value = MockRekaStreamResponse(
            "Mocked async streaming response about a cat image"
        )

        messages = [ChatMessage(role=MessageRole.USER, content="What's in this image?")]
        stream = await mock_reka_mm_llm.astream_chat(
            messages, image_documents=[mock_image_document]
        )

        response = [chunk async for chunk in stream]
        assert all(isinstance(chunk, ChatResponse) for chunk in response)
        assert [chunk.message.content for chunk in response] == [
            "Mocked",
            "async",
            "streaming",
            "response",
            "about",
            "a",
            "cat",
            "image",
        ]
        mock_acreate_stream.assert_called_once()


@pytest.mark.asyncio
async def test_astream_complete_mock_with_image(mock_reka_mm_llm, mock_image_document):
    with mock.patch.object(
        mock_reka_mm_llm._aclient.chat, "create_stream"
    ) as mock_acreate_stream:
        mock_acreate_stream.return_value = MockRekaStreamResponse(
            "Mocked async streaming completion about a cat image"
        )

        stream = await mock_reka_mm_llm.astream_complete(
            "Describe this image:", image_documents=[mock_image_document]
        )

        response = [chunk async for chunk in stream]
        assert all(isinstance(chunk, CompletionResponse) for chunk in response)
        assert [chunk.text for chunk in response] == [
            "Mocked",
            "async",
            "streaming",
            "completion",
            "about",
            "a",
            "cat",
            "image",
        ]
        mock_acreate_stream.assert_called_once()
