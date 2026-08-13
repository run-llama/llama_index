import json

import httpx
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.nebius import NebiusLLM
from llama_index.llms.nebius.base import DEFAULT_API_BASE


def test_llm_class() -> None:
    names_of_base_classes = [base.__name__ for base in NebiusLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_chat_uses_token_factory_endpoint_and_openai_request_shape() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == f"{DEFAULT_API_BASE}/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"

        payload = json.loads(request.content)
        assert payload["model"] == "Qwen/Qwen3-30B-A3B-fast"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["stream"] is False

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    llm = NebiusLLM(
        model="Qwen/Qwen3-30B-A3B-fast",
        api_key="test-key",
        http_client=http_client,
    )

    response = llm.chat([ChatMessage(role=MessageRole.USER, content="Hello")])

    assert response.message.content == "Hi!"
