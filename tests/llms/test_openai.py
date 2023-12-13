import os
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.llms.openai import OpenAI
from llama_index.llms.types import ChatMessage
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.completion import Completion, CompletionChoice, CompletionUsage

from tests.conftest import CachedOpenAIApiKeys


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


def mock_completion_v1(*args: Any, **kwargs: Any) -> Completion:
    return Completion(
        id="cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        object="text_completion",
        created=1589478378,
        model="text-davinci-003",
        choices=[
            CompletionChoice(
                text="\n\nThis is indeed a test",
                index=0,
                logprobs=None,
                finish_reason="length",
            )
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


async def mock_async_completion(*args: Any, **kwargs: Any) -> dict:
    return mock_completion(*args, **kwargs)


async def mock_async_completion_v1(*args: Any, **kwargs: Any) -> Completion:
    return mock_completion_v1(*args, **kwargs)


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


def mock_chat_completion_v1(*args: Any, **kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="gpt-3.5-turbo-0301",
        usage=CompletionUsage(prompt_tokens=13, completion_tokens=7, total_tokens=20),
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="\n\nThis is a test!"
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


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
    yield from responses


def mock_completion_stream_v1(
    *args: Any, **kwargs: Any
) -> Generator[Completion, None, None]:
    responses = [
        Completion(
            id="cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            object="text_completion",
            created=1589478378,
            model="text-davinci-003",
            choices=[CompletionChoice(text="1", finish_reason="stop", index=0)],
        ),
        Completion(
            id="cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            object="text_completion",
            created=1589478378,
            model="text-davinci-003",
            choices=[CompletionChoice(text="2", finish_reason="stop", index=0)],
        ),
    ]
    yield from responses


async def mock_async_completion_stream(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[dict, None]:
    async def gen() -> AsyncGenerator[dict, None]:
        for response in mock_completion_stream(*args, **kwargs):
            yield response

    return gen()


async def mock_async_completion_stream_v1(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[Completion, None]:
    async def gen() -> AsyncGenerator[Completion, None]:
        for response in mock_completion_stream_v1(*args, **kwargs):
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
    yield from responses


def mock_chat_completion_stream_v1(
    *args: Any, **kwargs: Any
) -> Generator[ChatCompletionChunk, None, None]:
    responses = [
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="\n\n"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(delta=ChoiceDelta(content="2"), finish_reason=None, index=0)
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[ChunkChoice(delta=ChoiceDelta(), finish_reason="stop", index=0)],
        ),
    ]
    yield from responses


@patch("llama_index.llms.openai.SyncOpenAI")
def test_completion_model_basic(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.completions.create.return_value = mock_completion_v1()

        llm = OpenAI(model="text-davinci-003")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text == "\n\nThis is indeed a test"

        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is indeed a test"


@patch("llama_index.llms.openai.SyncOpenAI")
def test_chat_model_basic(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion_v1()

        llm = OpenAI(model="gpt-3.5-turbo")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text == "\n\nThis is a test!"

        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is a test!"


@patch("llama_index.llms.openai.SyncOpenAI")
def test_completion_model_streaming(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.completions.create.return_value = mock_completion_stream_v1()

        llm = OpenAI(model="text-davinci-003")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response_gen = llm.stream_complete(prompt)
        responses = list(response_gen)
        assert responses[-1].text == "12"

        mock_instance.completions.create.return_value = mock_completion_stream_v1()
        chat_response_gen = llm.stream_chat([message])
        chat_responses = list(chat_response_gen)
        assert chat_responses[-1].message.content == "12"


@patch("llama_index.llms.openai.SyncOpenAI")
def test_chat_model_streaming(MockSyncOpenAI: MagicMock) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        mock_instance = MockSyncOpenAI.return_value
        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream_v1()
        )

        llm = OpenAI(model="gpt-3.5-turbo")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response_gen = llm.stream_complete(prompt)
        responses = list(response_gen)
        assert responses[-1].text == "\n\n2"

        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream_v1()
        )
        chat_response_gen = llm.stream_chat([message])
        chat_responses = list(chat_response_gen)
        assert chat_responses[-1].message.content == "\n\n2"
        assert chat_responses[-1].message.role == "assistant"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.AsyncOpenAI")
async def test_completion_model_async(MockAsyncOpenAI: MagicMock) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    create_fn = AsyncMock()
    create_fn.side_effect = mock_async_completion_v1
    mock_instance.completions.create = create_fn

    llm = OpenAI(model="text-davinci-003")
    prompt = "test prompt"
    message = ChatMessage(role="user", content="test message")

    response = await llm.acomplete(prompt)
    assert response.text == "\n\nThis is indeed a test"

    chat_response = await llm.achat([message])
    assert chat_response.message.content == "\n\nThis is indeed a test"


@pytest.mark.asyncio()
@patch("llama_index.llms.openai.AsyncOpenAI")
async def test_completion_model_async_streaming(MockAsyncOpenAI: MagicMock) -> None:
    mock_instance = MockAsyncOpenAI.return_value
    create_fn = AsyncMock()
    create_fn.side_effect = mock_async_completion_stream_v1
    mock_instance.completions.create = create_fn

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
        os.environ["OPENAI_API_KEY"] = "sk-" + ("a" * 48)

        # We can create a new LLM when the env variable is set
        assert OpenAI()

        os.environ["OPENAI_API_KEY"] = ""

        # We can create a new LLM when the api_key is set on the
        # class directly
        assert OpenAI(api_key="sk-" + ("a" * 48))
