import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llama_index.core.llms import ChatMessage
from llama_index.llms.maritalk import Maritalk

TEXT = "see data: here"
LINES = [f"data: {json.dumps({'text': TEXT})}", ""]


def test_stream_chat_keeps_data_prefix_inside_text():
    llm = Maritalk(api_key="test")
    response = MagicMock(ok=True)
    response.iter_lines.return_value = [line.encode() for line in LINES]
    with patch("requests.post", return_value=response):
        chunks = list(llm.stream_chat([ChatMessage(role="user", content="hi")]))
    assert [c.delta for c in chunks] == [TEXT]
    assert chunks[-1].message.content == TEXT


@pytest.mark.asyncio
async def test_astream_chat_keeps_data_prefix_inside_text():
    llm = Maritalk(api_key="test")
    body = "\n".join(LINES).encode()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    real_client = httpx.AsyncClient
    with patch("httpx.AsyncClient", lambda *a, **k: real_client(transport=transport)):
        gen = await llm.astream_chat([ChatMessage(role="user", content="hi")])
        chunks = [c async for c in gen]
    assert [c.delta for c in chunks] == [TEXT]
    assert chunks[-1].message.content == TEXT
