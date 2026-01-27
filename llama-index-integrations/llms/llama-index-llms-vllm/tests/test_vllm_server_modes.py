import json
from typing import Any, Dict, List, Optional

import pytest
import requests

from llama_index.llms.vllm import VllmServer


class FakeResponse:
    def __init__(
        self,
        json_data: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        iter_lines_data: Optional[List[bytes]] = None,
    ) -> None:
        self._json = json_data or {}
        self.status_code = status_code
        self._iter_lines = iter_lines_data or []
        self.content = json.dumps(self._json).encode("utf-8")

    def json(self) -> Dict[str, Any]:
        return self._json

    def iter_lines(self, **_: Any):
        for chunk in self._iter_lines:
            yield chunk

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_openai_like_complete(monkeypatch):
    recorded = {}

    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        recorded["url"] = url
        recorded["headers"] = headers
        recorded["json"] = json
        recorded["stream"] = stream
        recorded["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr("requests.post", fake_post)

    llm = VllmServer(
        api_url="http://mock/chat", openai_like=True, api_headers={"X-Test": "1"}
    )
    result = llm.complete("hi")

    assert result.text == "hello"
    assert recorded["url"] == "http://mock/chat"
    assert recorded["stream"] is False
    assert recorded["json"]["messages"][0]["content"] == "hi"
    assert recorded["headers"]["X-Test"] == "1"


def test_openai_like_stream(monkeypatch):
    chunks = [
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
        b"data: [DONE]\n",
    ]

    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        return FakeResponse(iter_lines_data=chunks, json_data={"choices": []})

    monkeypatch.setattr("requests.post", fake_post)

    llm = VllmServer(api_url="http://mock/chat", openai_like=True)
    outputs = list(llm.stream_complete("hi"))

    assert [o.delta for o in outputs] == ["he", "llo"]
    assert outputs[-1].text == "hello"


def test_native_http_error(monkeypatch):
    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        return FakeResponse(status_code=500)

    monkeypatch.setattr("requests.post", fake_post)

    llm = VllmServer(api_url="http://mock/native", openai_like=False)
    with pytest.raises(requests.HTTPError):
        llm.complete("hi")
