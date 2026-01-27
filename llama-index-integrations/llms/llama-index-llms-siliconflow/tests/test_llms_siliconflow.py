import json
import pytest
import types
from typing import Optional, Type
from unittest import mock
from requests import Response
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    CompletionResponse,
    ChatMessage,
    MessageRole,
)
from llama_index.llms.siliconflow import SiliconFlow

RESPONSE_JSON = {
    "id": "<string>",
    "choices": [
        {
            "message": {"role": "assistant", "content": "<string>"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 123,
        "completion_tokens": 123,
        "total_tokens": 123,
    },
    "created": 123,
    "model": "<string>",
    "object": "chat.completion",
}


class MockAsyncResponse:
    def __init__(self, json_data) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        pass

    async def __aenter__(self) -> "MockAsyncResponse":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[types.TracebackType],
    ) -> None:
        pass

    async def json(self) -> dict:
        return self._json_data


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in SiliconFlow.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_llm_model_alias():
    model = "deepseek-ai/DeepSeek-V2.5"
    api_key = "api_key_test"
    llm = SiliconFlow(model=model, api_key=api_key)
    assert llm.model == model
    assert llm.model_kwargs is not None


def test_llm_complete():
    input_text = "..."
    mock_response = Response()
    mock_response.status_code = 200
    mock_response._content = json.dumps(RESPONSE_JSON).encode("utf-8")
    expected_result = CompletionResponse(text="<string>", raw=RESPONSE_JSON)
    llm = SiliconFlow(api_key="...")
    with mock.patch("requests.Session.post", return_value=mock_response) as mock_post:
        actual_result = llm.complete(input_text)
        assert actual_result.text == expected_result.text
        assert actual_result.additional_kwargs == actual_result.additional_kwargs
        assert actual_result.raw == actual_result.raw
        assert actual_result.logprobs == actual_result.logprobs

        mock_post.assert_called_once_with(
            llm.base_url,
            json={
                "model": llm.model,
                "messages": [{"role": "user", "content": input_text}],
                "stream": False,
                "n": 1,
                "tools": None,
                "response_format": {"type": "text"},
                **llm.model_kwargs,
            },
            headers=llm._headers,
            timeout=llm.timeout,
        )


@pytest.mark.asyncio
async def test_llm_async_complete():
    input_text = "..."
    mock_response = MockAsyncResponse(json_data=RESPONSE_JSON)
    expected_result = CompletionResponse(text="<string>", raw=RESPONSE_JSON)
    llm = SiliconFlow(api_key="...")
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=mock_response
    ) as mock_post:
        actual_result = await llm.acomplete(input_text)

        assert actual_result.text == expected_result.text
        assert actual_result.additional_kwargs == actual_result.additional_kwargs
        assert actual_result.raw == actual_result.raw
        assert actual_result.logprobs == actual_result.logprobs

        mock_post.assert_called_once_with(
            llm.base_url,
            json={
                "model": llm.model,
                "messages": [{"role": "user", "content": input_text}],
                "stream": False,
                "n": 1,
                "tools": None,
                "response_format": {"type": "text"},
                **llm.model_kwargs,
            },
            headers=llm._headers,
            timeout=llm.timeout,
        )


def test_stream_chat():
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    # Create streaming response chunks in SSE format
    # Include a chunk with None content to test edge case
    stream_chunks = [
        b'data: {"id": "1", "choices": [{"delta": {"role": "assistant", "content": "Hello"}}]}\n',
        b'data: {"id": "2", "choices": [{"delta": {"content": " there"}}]}\n',
        b'data: {"id": "3", "choices": [{"delta": {"content": null}}]}\n',
        b'data: {"id": "4", "choices": [{"delta": {"content": "!"}}]}\n',
        b"data: [DONE]\n",
    ]

    # Create a mock response with iter_lines method
    mock_response = Response()
    mock_response.status_code = 200
    mock_response.iter_lines = mock.Mock(return_value=iter(stream_chunks))
    mock_response.raise_for_status = mock.Mock()

    llm = SiliconFlow(api_key="test_api_key")

    with mock.patch("requests.Session.post", return_value=mock_response) as mock_post:
        stream = llm.stream_chat(messages)

        # Collect all chunks from the stream
        responses = list(stream)

        # Verify we got responses (including the None content chunk)
        assert len(responses) == 4

        # Verify the first chunk
        assert responses[0].delta == "Hello"
        assert responses[0].message.content == "Hello"
        assert responses[0].message.role == MessageRole.ASSISTANT

        # Verify the second chunk
        assert responses[1].delta == " there"
        assert responses[1].message.content == "Hello there"

        # Verify the third chunk with None content (should be empty string)
        assert responses[2].delta == ""
        assert responses[2].message.content == "Hello there"

        # Verify the fourth chunk
        assert responses[3].delta == "!"
        assert responses[3].message.content == "Hello there!"

        # Verify the API was called correctly
        mock_post.assert_called_once_with(
            llm.base_url,
            json={
                "model": llm.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "n": 1,
                "tools": None,
                "response_format": {"type": "text"},
                **llm.model_kwargs,
            },
            headers=llm._headers,
            timeout=llm.timeout,
        )


@pytest.mark.asyncio
async def test_astream_chat():
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    # Create streaming response chunks in SSE format
    # Include a chunk with None content to test edge case
    stream_chunks = [
        b'data: {"id": "1", "choices": [{"delta": {"role": "assistant", "content": "Hi"}}]}\n',
        b'data: {"id": "2", "choices": [{"delta": {"content": " there"}}]}\n',
        b'data: {"id": "3", "choices": [{"delta": {"content": null}}]}\n',
        b'data: {"id": "4", "choices": [{"delta": {"content": "!"}}]}\n',
        b"data: [DONE]\n",
    ]

    # Create a mock async response with iter_any method
    class MockStreamContent:
        def __init__(self, chunks):
            self._chunks = chunks

        async def iter_any(self):
            for chunk in self._chunks:
                yield chunk

    class MockAsyncStreamResponse:
        def __init__(self, chunks):
            self.status = 200
            self.content = MockStreamContent(chunks)

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_response = MockAsyncStreamResponse(stream_chunks)

    llm = SiliconFlow(api_key="test_api_key")

    with mock.patch(
        "aiohttp.ClientSession.post", return_value=mock_response
    ) as mock_post:
        stream = await llm.astream_chat(messages)

        # Collect all chunks from the async stream
        responses = []
        async for response in stream:
            responses.append(response)

        # Verify we got responses (including the None content chunk)
        assert len(responses) == 4

        # Verify the first chunk
        assert responses[0].delta == "Hi"
        assert responses[0].message.content == "Hi"
        assert responses[0].message.role == MessageRole.ASSISTANT

        # Verify the second chunk
        assert responses[1].delta == " there"
        assert responses[1].message.content == "Hi there"

        # Verify the third chunk with None content (should be empty string)
        assert responses[2].delta == ""
        assert responses[2].message.content == "Hi there"

        # Verify the fourth chunk
        assert responses[3].delta == "!"
        assert responses[3].message.content == "Hi there!"

        # Verify the API was called correctly
        mock_post.assert_called_once_with(
            llm.base_url,
            json={
                "model": llm.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "n": 1,
                "tools": None,
                "response_format": {"type": "text"},
                **llm.model_kwargs,
            },
            headers=llm._headers,
            timeout=llm.timeout,
        )
