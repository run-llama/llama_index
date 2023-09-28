from typing import Any, AsyncGenerator, Generator


try:
    import litellm
except ImportError:
    litellm = None  # type: ignore

import pytest
from pytest import MonkeyPatch

from llama_index.llms.base import ChatMessage
from llama_index.llms.litellm import LiteLLM

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


@pytest.mark.skipif(litellm is None, reason="litellm not installed")
def test_chat_model_basic(monkeypatch: MonkeyPatch) -> None:
    with CachedOpenAIApiKeys(set_fake_key=True):
        monkeypatch.setattr(
            "llama_index.llms.litellm.completion_with_retry", mock_chat_completion
        )

        llm = LiteLLM(model="gpt-3.5-turbo")
        prompt = "test prompt"
        message = ChatMessage(role="user", content="test message")

        response = llm.complete(prompt)
        assert response.text == "\n\nThis is a test!"

        chat_response = llm.chat([message])
        assert chat_response.message.content == "\n\nThis is a test!"


@pytest.mark.skipif(litellm is None, reason="litellm not installed")
def test_metadata() -> None:
    llm = LiteLLM(model="gpt-3.5-turbo")
    assert isinstance(llm.metadata.context_window, int)

# Extra local LiteLLM Tests

# from llama_index.llms import LiteLLM, ChatMessage

# message = ChatMessage(role="user", content="why does LiteLLM love LlamaIndex")

# # # deep infra call
# os.environ["DEEPINFRA_API_KEY"] = "60UsYw6vmDkYMLVLIZ9vRzHhqOmibo0b" # api key
# os.environ["OPENAI_API_KEY"] = "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"

# llm = LiteLLM(model="gpt-3.5-turbo")
# chat_response = llm.chat([message])
# print("gpt-3.5-turbo Chat response\n")
# print(chat_response)
# ## streaming requests:
# # resp = llm.stream_chat([message])
# # print("\ngpt-3.5-turbo Streaming response\n")
# # for r in resp:
# #     print(r.delta, end="")


# ####### Deep infra  ######
# llm = LiteLLM(model="deepinfra/meta-llama/Llama-2-70b-chat-hf")
# message = ChatMessage(role="user", content="why does LiteLLM love LlamaIndex")
# chat_response = llm.chat([message])
# print("\ndeepinfra Chat response\n")
# print(chat_response)

# print(message)
# # resp = llm.stream_chat([message])
# # print("\ndeepinfra Streaming response\n")
# # for r in resp:
# #     print(r.delta, end="")

# os.environ["HUGGINGFACE_API_KEY"] = "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" # api key
# llm = LiteLLM(
#     model="huggingface/glaiveai/glaive-coder-7b", 
#     api_base="https://wjiegasee9bmqke2.us-east-1.aws.endpoints.huggingface.cloud"
# )
# message = ChatMessage(role="user", content="why does LiteLLM love LlamaIndex")
# chat_response = llm.chat([message])
# print("\nHF Chat response\n")
# print(chat_response)

# # resp = llm.stream_chat([message])
# # print("\nHF Streaming response\n")
# # for r in resp:
# #     print(r.delta, end="")