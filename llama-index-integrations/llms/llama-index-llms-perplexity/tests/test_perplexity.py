import inspect
import os
from collections.abc import AsyncGenerator, AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import requests
from tenacity import RetryError

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.llms.perplexity import Perplexity


@pytest.fixture()
def perplexity_llm():
    api_key = os.getenv("PPLX_API_KEY")
    if api_key is None:
        pytest.skip("PPLX_API_KEY not set in environment")
    return Perplexity(api_key=api_key)


@pytest.fixture()
def mock_perplexity_llm():
    return Perplexity(api_key="test")


def test_get_context_window():
    llm = Perplexity(api_key="dummy", model="sonar")
    assert llm._get_context_window() == 127072
    llm.model = "sonar-pro"
    assert llm._get_context_window() == 200000


def test_get_all_kwargs():
    llm = Perplexity(api_key="dummy", additional_kwargs={"foo": "bar"}, temperature=0.7)
    all_kwargs = llm._get_all_kwargs(custom=123)
    assert all_kwargs["foo"] == "bar"
    assert all_kwargs["custom"] == 123
    assert all_kwargs["temperature"] == 0.7


@pytest.mark.parametrize(
    ("headers", "expected"),
    [(None, "llamaindex"), ({"x-pplx-integration": "custom"}, "custom")],
)
def test_integration_header_can_be_overridden(headers, expected):
    llm = Perplexity(api_key="dummy", headers=headers)
    sync_request = requests.Request(
        "POST", "https://api.perplexity.ai", headers=llm.headers
    ).prepare()
    async_request = httpx.Request(
        "POST", "https://api.perplexity.ai", headers=llm.headers
    )

    assert sync_request.headers["x-pplx-integration"] == expected
    assert async_request.headers.get_list("x-pplx-integration") == [expected]


def test_chat(perplexity_llm):
    messages = [
        ChatMessage(role="system", content="Be precise and concise."),
        ChatMessage(role="user", content="Tell me 5 sentences about Perplexity."),
    ]
    response = perplexity_llm.chat(messages)
    assert isinstance(response, ChatResponse)
    assert response.message.content.strip()


def test_complete(perplexity_llm):
    prompt = "Perplexity is a company that provides"
    response = perplexity_llm.complete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip()


def test_stream_chat(perplexity_llm):
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(
            role="user", content="Name the first 5 elements in the periodic table."
        ),
    ]
    stream = perplexity_llm.stream_chat(messages)
    assert inspect.isgenerator(stream)
    response = ""
    for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


def test_stream_complete(perplexity_llm):
    prompt = "List the first 5 planets in the solar system:"
    stream = perplexity_llm.stream_complete(prompt)
    assert inspect.isgenerator(stream)
    response = ""
    for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


@pytest.mark.asyncio
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
    assert response.message.content.strip()


@pytest.mark.asyncio
async def test_acomplete(perplexity_llm):
    prompt = "The largest planet in our solar system is"
    response = await perplexity_llm.acomplete(prompt)
    assert isinstance(response, CompletionResponse)
    assert response.text.strip()


@pytest.mark.asyncio
async def test_astream_chat(perplexity_llm):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER,
            content="Name the first 5 elements in the periodic table.",
        ),
    ]
    stream = await perplexity_llm.astream_chat(messages)
    assert isinstance(stream, AsyncIterator)
    response = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


@pytest.mark.asyncio
async def test_astream_complete(perplexity_llm):
    prompt = "List the first 5 elements in the periodic table:"
    stream = await perplexity_llm.astream_complete(prompt)
    assert isinstance(stream, AsyncIterator)
    response = ""
    async for chunk in stream:
        assert isinstance(chunk, CompletionResponse)
        assert chunk.delta is not None
        response += chunk.delta
    assert response.strip()


def test_chat_mock(mock_perplexity_llm):
    with patch.object(
        mock_perplexity_llm,
        "_chat",
        return_value=ChatResponse(
            message=ChatMessage(role="assistant", content="mock")
        ),
    ) as mock_chat:
        messages = [ChatMessage(role="user", content="Hi")]
        result = mock_perplexity_llm.chat(messages)
        assert result.message.content == "mock"
        mock_chat.assert_called_once_with(messages)


def test_complete_mock(mock_perplexity_llm):
    with patch.object(
        mock_perplexity_llm, "_complete", return_value=CompletionResponse(text="mock")
    ) as mock_complete:
        result = mock_perplexity_llm.complete("hello")
        assert result.text == "mock"
        mock_complete.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_achat_mock(mock_perplexity_llm):
    with patch.object(
        mock_perplexity_llm,
        "_achat",
        new=AsyncMock(
            return_value=ChatResponse(
                message=ChatMessage(role="assistant", content="mock")
            )
        ),
    ) as mock_achat:
        messages = [ChatMessage(role="user", content="Hi")]
        result = await mock_perplexity_llm.achat(messages)
        assert result.message.content == "mock"
        mock_achat.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_acomplete_mock(mock_perplexity_llm):
    with patch.object(
        mock_perplexity_llm,
        "_acomplete",
        new=AsyncMock(return_value=CompletionResponse(text="mock")),
    ) as mock_acomplete:
        result = await mock_perplexity_llm.acomplete("hello")
        assert result.text == "mock"
        mock_acomplete.assert_called_once_with("hello")


def test_stream_chat_mock(mock_perplexity_llm):
    def mock_generator():
        yield ChatResponse(
            message=ChatMessage(role="assistant", content="Hello world"), delta="Hello "
        )
        yield ChatResponse(
            message=ChatMessage(role="assistant", content="Hello world"), delta="world"
        )

    with patch.object(
        mock_perplexity_llm,
        "_stream_chat",
        return_value=mock_generator(),
    ) as mock_stream:
        messages = [ChatMessage(role="user", content="Hi")]
        result = "".join(
            chunk.delta for chunk in mock_perplexity_llm.stream_chat(messages)
        )
        assert result == "Hello world"
        mock_stream.assert_called_once_with(messages)


def test_stream_complete_mock(mock_perplexity_llm):
    def mock_generator():
        yield CompletionResponse(text="Hi there", delta="Hi ")
        yield CompletionResponse(text="Hi there", delta="there")

    with patch.object(
        mock_perplexity_llm,
        "_stream_complete",
        return_value=mock_generator(),
    ) as mock_stream:
        result = "".join(
            chunk.delta for chunk in mock_perplexity_llm.stream_complete("Yo")
        )
        assert result == "Hi there"
        mock_stream.assert_called_once_with("Yo")


@pytest.mark.asyncio
async def test_astream_chat_mock(mock_perplexity_llm):
    messages = [
        ChatMessage(role=MessageRole.USER, content="Test message 1"),
        ChatMessage(role=MessageRole.USER, content="Test message 2"),
    ]

    async def ret_agen(*args, **kwargs):
        items = ["Mocked ", "streamed ", "chat"]
        content = ""
        for delta in items:
            content += delta
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content,
                ),
                delta=delta,
            )

    mock_astream_chat = AsyncMock(return_value=ret_agen())

    with patch.object(mock_perplexity_llm, "_astream_chat", mock_astream_chat):
        stream = await mock_perplexity_llm.astream_chat(messages)
        assert isinstance(stream, AsyncGenerator)
        full_response = ""
        async for each in stream:
            full_response += each.delta

        assert full_response == "Mocked streamed chat"
        mock_astream_chat.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_astream_complete_mock(mock_perplexity_llm):
    async def ret_agen(*args, **kwargs):
        yield CompletionResponse(text="Mocked ", delta="Mocked ")
        yield CompletionResponse(text="Mocked streamed ", delta="streamed ")
        yield CompletionResponse(text="Mocked streamed completion", delta="completion")

    prompt = "Test prompt"
    mock_astream_complete = AsyncMock(return_value=ret_agen())

    with patch.object(mock_perplexity_llm, "_astream_complete", mock_astream_complete):
        stream = await mock_perplexity_llm.astream_complete(prompt)
        assert isinstance(stream, AsyncGenerator)
        full_response = ""
        async for each in stream:
            full_response += each.delta
        assert full_response == "Mocked streamed completion"
        mock_astream_complete.assert_called_once_with(prompt)


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("complete", ("test",)),
        ("chat", ([ChatMessage(role=MessageRole.USER, content="Hi")],)),
    ],
)
def test_sync_errors_are_not_retried(mock_perplexity_llm, method, args):
    with patch.object(
        mock_perplexity_llm, f"_{method}", side_effect=RuntimeError("boom")
    ) as inner:
        with pytest.raises(RuntimeError):
            getattr(mock_perplexity_llm, method)(*args)
    assert inner.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_sent_once(mock_perplexity_llm, status):
    with patch.object(
        mock_perplexity_llm, "_chat", side_effect=_http_error(status)
    ) as inner:
        with pytest.raises(requests.HTTPError):
            mock_perplexity_llm.chat([ChatMessage(role=MessageRole.USER, content="Hi")])
    assert inner.call_count == 1


@pytest.mark.parametrize("status", [429, 503])
def test_transient_errors_are_retried(status):
    llm = Perplexity(api_key="test", max_retries=3)
    with patch.object(llm, "_complete", side_effect=_http_error(status)) as inner:
        with pytest.raises(RetryError) as e:
            llm.complete("test")
    assert inner.call_count == 3
    assert isinstance(e.value.__cause__, requests.HTTPError)


def test_connection_error_then_success_is_retried():
    llm = Perplexity(api_key="test", max_retries=3)
    ok = CompletionResponse(text="ok")
    with patch.object(
        llm, "_complete", side_effect=[requests.ConnectionError("reset"), ok]
    ) as inner:
        assert llm.complete("test").text == "ok"
    assert inner.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("acomplete", ("test",)),
        ("achat", ([ChatMessage(role=MessageRole.USER, content="Hi")],)),
    ],
)
async def test_async_errors_are_not_retried(mock_perplexity_llm, method, args):
    inner = AsyncMock(side_effect=ValueError("fail"))
    with patch.object(mock_perplexity_llm, f"_{method}", new=inner):
        with pytest.raises(ValueError):
            await getattr(mock_perplexity_llm, method)(*args)
    assert inner.await_count == 1


@pytest.mark.asyncio
async def test_async_client_error_is_sent_once(mock_perplexity_llm):
    request = httpx.Request("POST", "https://api.perplexity.ai/chat/completions")
    error = httpx.HTTPStatusError(
        "401", request=request, response=httpx.Response(401, request=request)
    )
    inner = AsyncMock(side_effect=error)
    with patch.object(mock_perplexity_llm, "_achat", new=inner):
        with pytest.raises(httpx.HTTPStatusError):
            await mock_perplexity_llm.achat(
                [ChatMessage(role=MessageRole.USER, content="Hi")]
            )
    assert inner.await_count == 1
