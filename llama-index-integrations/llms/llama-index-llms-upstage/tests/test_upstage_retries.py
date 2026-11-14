"""The decorator must not multiply a single failing call.

Counts are read off an httpx.MockTransport that answers every request with 429 and
`Retry-After: 0`. They hold for both parent versions of `llama-index-llms-openai`:
before #23942 the SDK still retries (1 + max_retries requests), after it the client
is created with `max_retries=0` and one fewer request is made.
"""

import httpx
import openai
import pytest

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.upstage import Upstage


def _429() -> httpx.Response:
    return httpx.Response(
        429,
        headers={"Retry-After": "0"},
        json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
    )


def _chat_ok() -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "solar-pro",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _sync_llm(store, max_retries: int) -> Upstage:
    def handler(request: httpx.Request) -> httpx.Response:
        store.append(request)
        return _429()

    return Upstage(
        model="solar-pro",
        api_key="fake",
        max_retries=max_retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_chat_is_not_multiplied() -> None:
    requests: list[httpx.Request] = []
    llm = _sync_llm(requests, max_retries=1)
    with pytest.raises(openai.RateLimitError):
        llm.chat([ChatMessage(role=MessageRole.USER, content="hi")])
    assert len(requests) <= 2, f"max_retries=1 produced {len(requests)} requests"


@pytest.mark.asyncio()
async def test_achat_is_not_multiplied() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _429()

    llm = Upstage(
        model="solar-pro",
        api_key="fake",
        max_retries=1,
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(openai.RateLimitError):
        await llm.achat([ChatMessage(role=MessageRole.USER, content="hi")])
    assert len(requests) <= 2, f"max_retries=1 produced {len(requests)} requests"


def test_stream_chat_keeps_its_retries() -> None:
    requests: list[httpx.Request] = []
    llm = _sync_llm(requests, max_retries=1)
    with pytest.raises(openai.RateLimitError):
        list(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="hi")]))
    assert 1 <= len(requests) <= 2, f"streaming made {len(requests)} requests"


@pytest.mark.asyncio()
async def test_async_chat_still_returns_a_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_ok())

    llm = Upstage(
        model="solar-pro",
        api_key="fake",
        max_retries=1,
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await llm.achat([ChatMessage(role=MessageRole.USER, content="hi")])
    assert response.message.content == "ok"
