from pytest import MonkeyPatch
from typing import Any
from llama_index.llms.base import ChatMessage
from llama_index.llms.openai import OpenAI


def mock_completion_with_retry(*args: Any, **kwargs: Any):
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
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


def mock_chat_completion_with_retry(*args: Any, **kwargs: Any):
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {"role": "assistant", "content": "\n\nThis is a test!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def test_completion_model_basic(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        "llama_index.llms.openai.completion_with_retry", mock_completion_with_retry
    )
    llm = OpenAI(model="text-davinci-003")

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.complete(prompt)
    llm.stream_complete(prompt)

    llm.chat([message])
    llm.stream_chat([message])


def test_chat_model_basic(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        "llama_index.llms.openai.completion_with_retry", mock_chat_completion_with_retry
    )
    llm = OpenAI(model="gpt-3.5-turbo")

    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    llm.complete(prompt)
    llm.stream_complete(prompt)

    llm.chat([message])
    llm.stream_chat([message])
