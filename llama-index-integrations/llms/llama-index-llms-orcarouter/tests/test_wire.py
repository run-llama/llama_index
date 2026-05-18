import json

import httpx
import respx
from llama_index.core.llms import ChatMessage
from llama_index.llms.orcarouter import OrcaRouter

_RESP = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "orcarouter/auto",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@respx.mock
def test_request_shape_default() -> None:
    route = respx.post("https://api.orcarouter.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_RESP)
    )
    llm = OrcaRouter(api_key="sk-test")
    llm.chat([ChatMessage(role="user", content="hi")])

    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer sk-test"
    assert req.headers["http-referer"] == "https://www.llamaindex.ai/"
    assert req.headers["x-title"] == "LlamaIndex"

    body = json.loads(req.content)
    assert body["model"] == "orcarouter/auto"
    assert "extra_body" not in body  # no fallback => no extra_body emitted


@respx.mock
def test_request_shape_with_fallback() -> None:
    route = respx.post("https://api.orcarouter.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_RESP)
    )
    llm = OrcaRouter(
        api_key="sk-test",
        model="openai/gpt-4o-mini",
        fallback_models=["openai/gpt-4o", "anthropic/claude-sonnet-4.6"],
    )
    llm.chat([ChatMessage(role="user", content="hi")])

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "openai/gpt-4o-mini"
    # OpenAI SDK flattens extra_body into the top-level request body
    assert body["models"] == ["openai/gpt-4o", "anthropic/claude-sonnet-4.6"]
    assert body["route"] == "fallback"


@respx.mock
def test_attribution_user_override_on_wire() -> None:
    route = respx.post("https://api.orcarouter.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_RESP)
    )
    llm = OrcaRouter(
        api_key="sk-test",
        default_headers={"X-Title": "my-app"},
    )
    llm.chat([ChatMessage(role="user", content="hi")])

    req = route.calls.last.request
    assert req.headers["x-title"] == "my-app"
    assert req.headers["http-referer"] == "https://www.llamaindex.ai/"
