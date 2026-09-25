"""Custom ``http_client`` pass-through, across both openai major versions.

openai 3 migrated to HTTPX2 and stopped installing ``httpx`` itself, so the
``http_client``/``async_http_client`` arguments may now be handed either an
``httpx`` or an ``httpx2`` client. Both must reach the wire. These tests run
against a local stdlib HTTP server so they need no API key and no network.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Iterator, List, Tuple

import httpx
import pytest

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

try:
    import httpx2
except ImportError:  # openai<3 does not install httpx2
    httpx2 = None

_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "pong"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_CHUNK = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "pong"}}],
}


SEEN_HEADERS: List[Any] = []
MARKER = "x-llamaindex-custom-client"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        SEEN_HEADERS.append(dict(self.headers))
        raw = self.rfile.read(int(self.headers.get("content-length", 0)) or 0)
        streaming = json.loads(raw or b"{}").get("stream", False)
        if streaming:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps(_CHUNK)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps(_COMPLETION).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr request logging."""


@pytest.fixture(scope="module")
def api_base() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _sync_clients() -> List[Tuple[str, Any]]:
    clients = [("httpx", httpx.Client)]
    if httpx2 is not None:
        clients.append(("httpx2", httpx2.Client))
    return clients


def _async_clients() -> List[Tuple[str, Any]]:
    clients = [("httpx", httpx.AsyncClient)]
    if httpx2 is not None:
        clients.append(("httpx2", httpx2.AsyncClient))
    return clients


def _llm(api_base: str, **kwargs: Any) -> OpenAI:
    return OpenAI(model="gpt-4o-mini", api_key="sk-test", api_base=api_base, **kwargs)


@pytest.mark.parametrize(("name", "client_cls"), _sync_clients())
def test_custom_http_client_chat(api_base: str, name: str, client_cls: Any) -> None:
    SEEN_HEADERS.clear()
    with client_cls(headers={MARKER: name}) as client:
        response = _llm(api_base, http_client=client).chat(
            [ChatMessage(role="user", content="ping")]
        )
    assert response.message.content == "pong"
    # proves the supplied client carried the request, not the SDK default
    assert SEEN_HEADERS and SEEN_HEADERS[-1].get(MARKER) == name


@pytest.mark.parametrize(("name", "client_cls"), _sync_clients())
def test_custom_http_client_stream(api_base: str, name: str, client_cls: Any) -> None:
    SEEN_HEADERS.clear()
    with client_cls(headers={MARKER: name}) as client:
        chunks = list(
            _llm(api_base, http_client=client).stream_chat(
                [ChatMessage(role="user", content="ping")]
            )
        )
    assert chunks
    assert chunks[-1].message.content == "pong"
    assert SEEN_HEADERS and SEEN_HEADERS[-1].get(MARKER) == name


@pytest.mark.parametrize(("name", "client_cls"), _async_clients())
def test_custom_async_http_client_chat(
    api_base: str, name: str, client_cls: Any
) -> None:
    SEEN_HEADERS.clear()

    async def go() -> Any:
        async with client_cls(headers={MARKER: name}) as client:
            return await _llm(api_base, async_http_client=client).achat(
                [ChatMessage(role="user", content="ping")]
            )

    assert asyncio.run(go()).message.content == "pong"
    assert SEEN_HEADERS and SEEN_HEADERS[-1].get(MARKER) == name


def test_default_client_needs_no_explicit_http_client(api_base: str) -> None:
    """The SDK's own default transport still works when nothing is passed."""
    response = _llm(api_base).chat([ChatMessage(role="user", content="ping")])
    assert response.message.content == "pong"
