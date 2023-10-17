from typing import Any, Generator

import pytest
from llama_index.llms.base import ChatMessage
from llama_index.llms.konko import Konko
from pytest import MonkeyPatch

try:
    import konko
except ImportError:
    konko = None  # type: ignore


def setup_module() -> None:
    import os

    os.environ["KONKO_API_KEY"] = "ko-" + "a" * 48


def mock_chat_completion(*args: Any, **kwargs: Any) -> dict:
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "meta-llama/Llama-2-13b-chat-hf",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {"role": "assistant", "content": "\n\nThis is a test!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def mock_completion(*args: Any, **kwargs: Any) -> dict:
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "meta-llama/Llama-2-13b-chat-hf",
        "choices": [
            {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def mock_chat_completion_stream(
    *args: Any, **kwargs: Any
) -> Generator[dict, None, None]:
    # Example taken from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
    responses = [
        {
            "choices": [
                {"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "meta-llama/Llama-2-13b-chat-hf",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [
                {"delta": {"content": "\n\n"}, "finish_reason": None, "index": 0}
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [{"delta": {"content": "2"}, "finish_reason": None, "index": 0}],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
    ]
    yield from responses


@pytest.mark.skipif(konko is None, reason="konko not installed")
def test_chat_model_basic_non_openai_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.konko.completion_with_retry", mock_chat_completion
    )
    llm = Konko(model="meta-llama/Llama-2-13b-chat-hf")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response = llm.complete(prompt)
    assert response.text is not None

    chat_response = llm.chat([message])
    assert chat_response.message.content is not None


@pytest.mark.skipif(konko is None, reason="konko not installed")
def test_chat_model_basic_openai_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.konko.completion_with_retry", mock_chat_completion
    )
    llm = Konko(model="gpt-3.5-turbo")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response = llm.complete(prompt)
    assert response.text is not None

    chat_response = llm.chat([message])
    assert chat_response.message.content is not None


@pytest.mark.skipif(konko is None, reason="konko not installed")
def test_chat_model_streaming(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.konko.completion_with_retry", mock_chat_completion_stream
    )
    llm = Konko(model="meta-llama/Llama-2-13b-chat-hf")
    message = ChatMessage(role="user", content="test message")
    chat_response_gen = llm.stream_chat([message])
    chat_responses = list(chat_response_gen)
    assert chat_responses[-1].message.content is not None


def teardown_module() -> None:
    import os

    del os.environ["KONKO_API_KEY"]
