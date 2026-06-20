"""Regression tests for the async streaming session lifecycle.

`_astream_complete` / `_astream_chat` must read the streamed response body while
the ``aiohttp.ClientSession`` is still open. Previously they returned the
response out of the ``async with aiohttp.ClientSession()`` block, so the session
was closed before ``response.content`` was iterated, raising at read time.
"""

import aiohttp
import pytest

from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.perplexity import Perplexity


class _FakeContent:
    """Async line iterator that fails once its session has been closed,
    mirroring aiohttp's behaviour when the connection is released."""

    def __init__(self, lines, session):
        self._lines = list(lines)
        self._session = session

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._session.closed:
            raise RuntimeError("Cannot read from a closed aiohttp session")
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeResponse:
    def __init__(self, lines, session):
        self.content = _FakeContent(lines, session)

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def post(self, *args, **kwargs):
        return _FakeResponse(self._lines, self)


_SSE_LINES = [
    b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
    b'data: {"choices": [{"delta": {"content": " world"}}]}',
    b"data: [DONE]",
]


@pytest.mark.asyncio
async def test_astream_complete_reads_body_before_session_closes(monkeypatch):
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **k: _FakeSession(list(_SSE_LINES))
    )
    llm = Perplexity(api_key="test", model="sonar")
    gen = await llm.astream_complete("hi")
    deltas = [chunk.delta async for chunk in gen]
    assert "".join(deltas) == "Hello world"


@pytest.mark.asyncio
async def test_astream_chat_reads_body_before_session_closes(monkeypatch):
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **k: _FakeSession(list(_SSE_LINES))
    )
    llm = Perplexity(api_key="test", model="sonar")
    gen = await llm.astream_chat([ChatMessage(role="user", content="hi")])
    deltas = [chunk.delta async for chunk in gen]
    assert "".join(deltas) == "Hello world"
