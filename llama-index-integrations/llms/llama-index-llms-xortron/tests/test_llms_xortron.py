import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.xortron import Xortron

# ---------------------------------------------------------------------------
# Initialisation & metadata
# ---------------------------------------------------------------------------


def test_class_name():
    llm = Xortron(model="xortron-7b")
    assert llm.class_name() == "Xortron_llm"


def test_metadata():
    llm = Xortron(model="xortron-7b", context_window=4096, max_tokens=512)
    metadata = llm.metadata
    assert metadata.model_name == "xortron-7b"
    assert metadata.context_window == 4096
    assert metadata.num_output == 512
    assert metadata.is_chat_model is True


def test_default_init():
    llm = Xortron()
    assert llm.model == "xortron-default"
    assert llm.base_url == "http://localhost:8000"
    assert llm.temperature == 0.7
    assert llm.api_key is None


def test_custom_init():
    llm = Xortron(
        model="xortron-13b",
        base_url="http://custom:9000",
        temperature=0.5,
        api_key="test-key",
        additional_kwargs={"top_p": 0.9},
    )
    assert llm.model == "xortron-13b"
    assert llm.base_url == "http://custom:9000"
    assert llm.temperature == 0.5
    assert llm.api_key == "test-key"
    assert llm.additional_kwargs == {"top_p": 0.9}


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def test_get_headers_no_api_key():
    llm = Xortron()
    headers = llm._get_headers()
    assert headers == {"Content-Type": "application/json"}
    assert "Authorization" not in headers


def test_get_headers_with_api_key():
    llm = Xortron(api_key="my-key")
    headers = llm._get_headers()
    assert headers["Authorization"] == "Bearer my-key"


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_build_payload():
    llm = Xortron(model="xortron-7b", temperature=0.5, max_tokens=256)
    payload = llm._build_payload(prompt="Hello")
    assert payload["model"] == "xortron-7b"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 256
    assert payload["prompt"] == "Hello"


def test_build_payload_with_messages():
    llm = Xortron(model="xortron-7b")
    messages = [{"role": "user", "content": "Hello"}]
    payload = llm._build_payload(messages=messages)
    assert payload["messages"] == messages
    assert "prompt" not in payload


def test_build_payload_additional_kwargs():
    llm = Xortron(additional_kwargs={"top_p": 0.9, "stop": ["\n"]})
    payload = llm._build_payload(prompt="Hi")
    assert payload["top_p"] == 0.9
    assert payload["stop"] == ["\n"]


def test_build_payload_override_kwargs():
    llm = Xortron(model="xortron-7b", temperature=0.7)
    payload = llm._build_payload(prompt="Hi", temperature=0.1)
    # Explicit kwarg should override the field value
    assert payload["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def test_convert_messages():
    llm = Xortron()
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]
    converted = llm._convert_messages(messages)
    assert len(converted) == 2
    assert converted[0] == {"role": "system", "content": "You are helpful."}
    assert converted[1] == {"role": "user", "content": "Hello"}


def test_convert_messages_empty():
    llm = Xortron()
    assert llm._convert_messages([]) == []


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_completion_response():
    llm = Xortron()
    data = {"text": "Paris is the capital of France."}
    response = llm._parse_completion_response(data)
    assert response.text == "Paris is the capital of France."
    assert response.raw == data


def test_parse_completion_response_output_key():
    llm = Xortron()
    data = {"output": "Paris is the capital."}
    response = llm._parse_completion_response(data)
    assert response.text == "Paris is the capital."


def test_parse_completion_response_completion_key():
    llm = Xortron()
    data = {"completion": "Paris."}
    response = llm._parse_completion_response(data)
    assert response.text == "Paris."


def test_parse_completion_response_empty():
    llm = Xortron()
    data = {"some_other_key": "value"}
    response = llm._parse_completion_response(data)
    assert response.text == ""


def test_parse_chat_response():
    llm = Xortron()
    data = {"message": {"role": "assistant", "content": "Hello!"}}
    response = llm._parse_chat_response(data)
    assert response.message.content == "Hello!"
    assert response.raw == data


def test_parse_chat_response_flat():
    llm = Xortron()
    data = {"content": "Hello!", "role": "assistant"}
    response = llm._parse_chat_response(data)
    assert response.message.content == "Hello!"


def test_parse_chat_response_output_key():
    llm = Xortron()
    data = {"output": "Hello!"}
    response = llm._parse_chat_response(data)
    assert response.message.content == "Hello!"


# ---------------------------------------------------------------------------
# Sync complete / chat
# ---------------------------------------------------------------------------


def test_complete():
    llm = Xortron(model="xortron-7b")
    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "Test response"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(llm, "_client", create=True) as mock_client:
        mock_client.post.return_value = mock_response
        response = llm.complete("Test prompt")
        assert response.text == "Test response"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/v1/completions"


def test_chat():
    llm = Xortron(model="xortron-7b")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"role": "assistant", "content": "Hi there!"}
    }
    mock_response.raise_for_status = MagicMock()

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    with patch.object(llm, "_client", create=True) as mock_client:
        mock_client.post.return_value = mock_response
        response = llm.chat(messages)
        assert response.message.content == "Hi there!"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/v1/chat"


# ---------------------------------------------------------------------------
# Async complete / chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acomplete():
    llm = Xortron(model="xortron-7b")
    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "Async response"}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    llm._async_client = mock_async_client

    response = await llm.acomplete("Async prompt")
    assert response.text == "Async response"
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    assert call_args[0][0] == "/v1/completions"


@pytest.mark.asyncio
async def test_achat():
    llm = Xortron(model="xortron-7b")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"role": "assistant", "content": "Async hi!"}
    }
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    llm._async_client = mock_async_client

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    response = await llm.achat(messages)
    assert response.message.content == "Async hi!"
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    assert call_args[0][0] == "/v1/chat"


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _make_stream_context(lines):
    """Create a mock context manager that yields lines from iter_lines()."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)

    @contextmanager
    def stream_cm(*args, **kwargs):
        yield mock_resp

    return stream_cm


def _make_async_stream_context(lines):
    """Create a mock callable that returns an async context manager yielding lines."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    async def aiter_lines():
        for line in lines:
            yield line

    mock_resp.aiter_lines = aiter_lines

    class AsyncCtx:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, *args):
            pass

    def factory(*args, **kwargs):
        return AsyncCtx()

    return factory


# ---------------------------------------------------------------------------
# Sync streaming
# ---------------------------------------------------------------------------


def test_stream_complete():
    llm = Xortron(model="xortron-7b")

    sse_lines = [
        'data: {"text": "Hello"}',
        'data: {"text": " world"}',
        "data: [DONE]",
    ]

    mock_client = MagicMock()
    mock_client.stream = _make_stream_context(sse_lines)
    llm._client = mock_client

    chunks = list(llm.stream_complete("Say hello"))
    assert len(chunks) == 2
    assert chunks[0].delta == "Hello"
    assert chunks[0].text == "Hello"
    assert chunks[1].delta == " world"
    assert chunks[1].text == "Hello world"


def test_stream_complete_no_data_prefix():
    llm = Xortron(model="xortron-7b")

    # Lines without "data: " prefix (plain JSON lines)
    sse_lines = [
        json.dumps({"text": "A"}),
        json.dumps({"text": "B"}),
    ]

    mock_client = MagicMock()
    mock_client.stream = _make_stream_context(sse_lines)
    llm._client = mock_client

    chunks = list(llm.stream_complete("prompt"))
    assert len(chunks) == 2
    assert chunks[1].text == "AB"


def test_stream_complete_empty_and_done_lines():
    llm = Xortron(model="xortron-7b")

    sse_lines = [
        "",
        'data: {"text": "only"}',
        "",
        "data: [DONE]",
    ]

    mock_client = MagicMock()
    mock_client.stream = _make_stream_context(sse_lines)
    llm._client = mock_client

    chunks = list(llm.stream_complete("prompt"))
    assert len(chunks) == 1
    assert chunks[0].text == "only"


def test_stream_complete_malformed_json():
    llm = Xortron(model="xortron-7b")

    sse_lines = [
        'data: {"text": "good"}',
        "data: {bad json",
        'data: {"text": "ok"}',
    ]

    mock_client = MagicMock()
    mock_client.stream = _make_stream_context(sse_lines)
    llm._client = mock_client

    chunks = list(llm.stream_complete("prompt"))
    # Malformed line is skipped
    assert len(chunks) == 2
    assert chunks[0].text == "good"
    assert chunks[1].text == "goodok"


def test_stream_chat():
    llm = Xortron(model="xortron-7b")

    sse_lines = [
        'data: {"text": "Hi"}',
        'data: {"text": " there"}',
        "data: [DONE]",
    ]

    mock_client = MagicMock()
    mock_client.stream = _make_stream_context(sse_lines)
    llm._client = mock_client

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    chunks = list(llm.stream_chat(messages))
    assert len(chunks) == 2
    assert chunks[0].delta == "Hi"
    assert chunks[0].message.content == "Hi"
    assert chunks[1].delta == " there"
    assert chunks[1].message.content == "Hi there"
    assert chunks[1].message.role == MessageRole.ASSISTANT


def test_stream_chat_delta_key():
    llm = Xortron(model="xortron-7b")

    sse_lines = [
        'data: {"delta": "one"}',
        'data: {"delta": "two"}',
    ]

    mock_client = MagicMock()
    mock_client.stream = _make_stream_context(sse_lines)
    llm._client = mock_client

    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
    chunks = list(llm.stream_chat(messages))
    assert len(chunks) == 2
    assert chunks[1].message.content == "onetwo"


# ---------------------------------------------------------------------------
# Async streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_complete():
    llm = Xortron(model="xortron-7b")

    sse_lines = [
        'data: {"text": "Async"}',
        'data: {"text": " stream"}',
        "data: [DONE]",
    ]

    mock_async_client = MagicMock()
    mock_async_client.stream = _make_async_stream_context(sse_lines)
    llm._async_client = mock_async_client

    gen = await llm.astream_complete("prompt")
    chunks = [chunk async for chunk in gen]
    assert len(chunks) == 2
    assert chunks[0].delta == "Async"
    assert chunks[1].text == "Async stream"


@pytest.mark.asyncio
async def test_astream_chat():
    llm = Xortron(model="xortron-7b")

    sse_lines = [
        'data: {"text": "Hey"}',
        'data: {"text": "!"}',
        "data: [DONE]",
    ]

    mock_async_client = MagicMock()
    mock_async_client.stream = _make_async_stream_context(sse_lines)
    llm._async_client = mock_async_client

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    gen = await llm.astream_chat(messages)
    chunks = [chunk async for chunk in gen]
    assert len(chunks) == 2
    assert chunks[0].delta == "Hey"
    assert chunks[1].message.content == "Hey!"


@pytest.mark.asyncio
async def test_astream_complete_malformed_json():
    llm = Xortron(model="xortron-7b")

    sse_lines = [
        'data: {"text": "ok"}',
        "data: NOT_JSON",
        'data: {"text": "!"}',
    ]

    mock_async_client = MagicMock()
    mock_async_client.stream = _make_async_stream_context(sse_lines)
    llm._async_client = mock_async_client

    gen = await llm.astream_complete("prompt")
    chunks = [chunk async for chunk in gen]
    assert len(chunks) == 2
    assert chunks[1].text == "ok!"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_complete_http_error():
    llm = Xortron(model="xortron-7b")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    llm._client = mock_client

    with pytest.raises(httpx.HTTPStatusError):
        llm.complete("prompt")


def test_chat_http_error():
    llm = Xortron(model="xortron-7b")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found",
        request=MagicMock(),
        response=MagicMock(status_code=404),
    )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    llm._client = mock_client

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    with pytest.raises(httpx.HTTPStatusError):
        llm.chat(messages)


@pytest.mark.asyncio
async def test_acomplete_http_error():
    llm = Xortron(model="xortron-7b")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Gateway",
        request=MagicMock(),
        response=MagicMock(status_code=502),
    )

    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    llm._async_client = mock_async_client

    with pytest.raises(httpx.HTTPStatusError):
        await llm.acomplete("prompt")


@pytest.mark.asyncio
async def test_achat_http_error():
    llm = Xortron(model="xortron-7b")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized",
        request=MagicMock(),
        response=MagicMock(status_code=401),
    )

    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    llm._async_client = mock_async_client

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    with pytest.raises(httpx.HTTPStatusError):
        await llm.achat(messages)


def test_stream_complete_http_error():
    llm = Xortron(model="xortron-7b")

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )

    @contextmanager
    def stream_cm(*args, **kwargs):
        yield mock_resp

    mock_client = MagicMock()
    mock_client.stream = stream_cm
    llm._client = mock_client

    with pytest.raises(httpx.HTTPStatusError):
        list(llm.stream_complete("prompt"))


def test_stream_chat_http_error():
    llm = Xortron(model="xortron-7b")

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )

    @contextmanager
    def stream_cm(*args, **kwargs):
        yield mock_resp

    mock_client = MagicMock()
    mock_client.stream = stream_cm
    llm._client = mock_client

    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    with pytest.raises(httpx.HTTPStatusError):
        list(llm.stream_chat(messages))


# ---------------------------------------------------------------------------
# Client lazy initialisation
# ---------------------------------------------------------------------------


def test_client_lazy_init():
    llm = Xortron(base_url="http://localhost:9999", request_timeout=30.0)
    assert llm._client is None
    client = llm.client
    assert isinstance(client, httpx.Client)
    # Subsequent access returns the same instance
    assert llm.client is client


def test_async_client_lazy_init():
    llm = Xortron(base_url="http://localhost:9999", request_timeout=30.0)
    assert llm._async_client is None
    client = llm.async_client
    assert isinstance(client, httpx.AsyncClient)
    assert llm.async_client is client


# ---------------------------------------------------------------------------
# Request timeout
# ---------------------------------------------------------------------------


def test_custom_request_timeout():
    llm = Xortron(request_timeout=120.0)
    assert llm.request_timeout == 120.0


def test_default_request_timeout():
    llm = Xortron()
    assert llm.request_timeout == 60.0
