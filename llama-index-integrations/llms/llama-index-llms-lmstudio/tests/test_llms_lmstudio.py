import asyncio
import json
from typing import Any, List

import httpx
import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.lmstudio import LMStudio


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in LMStudio.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.fixture()
def captured_requests(monkeypatch: pytest.MonkeyPatch) -> List[httpx.Request]:
    requests = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if json.loads(request.content)["stream"]:
            event = {"choices": [{"delta": {"role": "assistant", "content": "ok"}}]}
            return httpx.Response(
                200,
                text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    transport = httpx.MockTransport(handle_request)
    sync_client = httpx.Client
    async_client = httpx.AsyncClient

    def make_sync_client(**kwargs: Any) -> httpx.Client:
        return sync_client(transport=transport, **kwargs)

    def make_async_client(**kwargs: Any) -> httpx.AsyncClient:
        return async_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", make_sync_client)
    monkeypatch.setattr(httpx, "AsyncClient", make_async_client)
    return requests


@pytest.mark.parametrize("method", ["chat", "achat", "stream_chat"])
def test_inference_parameters_use_openai_payload(
    method: str, captured_requests: List[httpx.Request]
) -> None:
    llm = LMStudio(
        model_name="test-model",
        temperature=0.0,
        num_output=32,
        context_window=8192,
        additional_kwargs={"top_p": 0.75, "seed": 42},
    )
    messages = [ChatMessage(role="user", content="Hello")]
    if method == "achat":
        response = asyncio.run(llm.achat(messages))
    elif method == "stream_chat":
        response = list(llm.stream_chat(messages))[-1]
    else:
        response = llm.chat(messages)

    assert response.message.content == "ok"
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert str(request.url) == "http://localhost:1234/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0,
        "max_tokens": 32,
        "top_p": 0.75,
        "seed": 42,
        "stream": method == "stream_chat",
    }
    assert llm.metadata.context_window == 8192


@pytest.mark.parametrize(
    ("call_kwargs", "expected_temperature", "expected_max_tokens"),
    [({}, 0.25, 48), ({"temperature": 0.0, "max_tokens": 16}, 0.0, 16)],
)
def test_inference_parameter_overrides(
    captured_requests: List[httpx.Request],
    call_kwargs: dict,
    expected_temperature: float,
    expected_max_tokens: int,
) -> None:
    llm = LMStudio(
        model_name="test-model",
        temperature=0.5,
        num_output=32,
        additional_kwargs={"temperature": 0.25, "max_tokens": 48},
    )
    llm.chat([ChatMessage(role="user", content="Hello")], **call_kwargs)
    payload = json.loads(captured_requests[0].content)
    assert payload["temperature"] == expected_temperature
    assert payload["max_tokens"] == expected_max_tokens
    assert "options" not in payload
