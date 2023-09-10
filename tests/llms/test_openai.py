import os
from typing import Any, AsyncGenerator, Generator

import openai
import pytest
from pytest import MonkeyPatch

from llama_index.llms.base import ChatMessage
from llama_index.llms.openai import OpenAI

from ..conftest import CachedOpenAIApiKeys


def mock_completion(*args: Any, **kwargs: Any) -> dict:
    # Example taken from https://platform.openai.com/docs/api-reference/completions/create
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


async def mock_async_completion(*args: Any, **kwargs: Any) -> dict:
    return mock_completion(*args, **kwargs)


def mock_chat_completion(*args: Any, **kwargs: Any) -> dict:
    # Example taken from https://platform.openai.com/docs/api-reference/chat/create
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


def mock_completion_stream(*args: Any, **kwargs: Any) -> Generator[dict, None, None]:
    # Example taken from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
    responses = [
        {
            "choices": [
                {
                    "text": "1",
                }
            ],
        },
        {
            "choices": [
                {
                    "text": "2",
                }
            ],
        },
    ]
    for response in responses:
        yield response


async def mock_async_completion_stream(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[dict, None]:
    async def gen() -> AsyncGenerator[dict, None]:
        for response in mock_completion_stream(*args, **kwargs):
            yield response

    return gen()


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
            "model": "gpt-3.5-turbo-0301",
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
    for response in responses:
        yield response


def test_completion_model_basic(monkeypatch: MonkeyPatch) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        monkeypatch.setattr(
            "llama_index.llms.openai.completion_with_retry", mock_completion
        )

        llm = OpenAI(model="text-davinci-003")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text == "\n\nThis is indeed a test"

        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is indeed a test"


def test_chat_model_basic(monkeypatch: MonkeyPatch) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        monkeypatch.setattr(
            "llama_index.llms.openai.completion_with_retry", mock_chat_completion
        )

        llm = OpenAI(model="gpt-3.5-turbo")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text == "\n\nThis is a test!"

        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is a test!"


def test_completion_model_streaming(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.openai.completion_with_retry", mock_completion_stream
    )

    llm = OpenAI(model="text-davinci-003")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response_gen = llm.stream_complete(prompt)
    responses = list(response_gen)
    assert responses[-1].text == "12"
    chat_response_gen = llm.stream_chat([message])
    chat_responses = list(chat_response_gen)
    assert chat_responses[-1].message.content == "12"


def test_chat_model_streaming(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.openai.completion_with_retry", mock_chat_completion_stream
    )

    llm = OpenAI(model="gpt-3.5-turbo")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response_gen = llm.stream_complete(prompt)
    responses = list(response_gen)
    assert responses[-1].text == "\n\n2"

    chat_response_gen = llm.stream_chat([message])
    chat_responses = list(chat_response_gen)
    assert chat_responses[-1].message.content == "\n\n2"
    assert chat_responses[-1].message.role == "assistant"


@pytest.mark.asyncio
async def test_completion_model_async(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.openai.acompletion_with_retry", mock_async_completion
    )

    llm = OpenAI(model="text-davinci-003")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response = await llm.acomplete(prompt)
    assert response.text == "\n\nThis is indeed a test"

    chat_response = await llm.achat([message])
    assert chat_response.message.content == "\n\nThis is indeed a test"


@pytest.mark.asyncio
async def test_completion_model_async_streaming(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.llms.openai.acompletion_with_retry",
        mock_async_completion_stream,
    )

    llm = OpenAI(model="text-davinci-003")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response_gen = await llm.astream_complete(prompt)
    responses = [item async for item in response_gen]
    assert responses[-1].text == "12"
    chat_response_gen = await llm.astream_chat([message])
    chat_responses = [item async for item in chat_response_gen]
    assert chat_responses[-1].message.content == "12"


def test_validates_api_key_is_present() -> None:
    with CachedOpenAIApiKeys():
        with pytest.raises(ValueError, match="No API key found for OpenAI."):
            OpenAI()

    os.environ["OPENAI_API_KEY"] = "sk-" + ("a" * 48)

    # We can create a new LLM when the env variable is set
    assert OpenAI()

    os.environ["OPENAI_API_KEY"] = ""
    openai.api_key = "sk-" + ("a" * 48)

    # We can create a new LLM when the api_key is set on the
    # library directly
    assert OpenAI()
