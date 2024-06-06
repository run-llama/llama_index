from unittest import mock

from ai21.models.chat import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionChunk,
    ChoicesChunk,
    ChoiceDelta,
    ChatMessage as AI21ChatMessage,
)
from ai21.models.usage_info import UsageInfo
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from llama_index.core.llms import ChatMessage

from llama_index.llms.ai21 import AI21

_FAKE_API_KEY = "fake-api-key"
_MODEL_NAME = "jamba-instruct"
_FAKE_CHAT_COMPLETIONS_RESPONSE = ChatCompletionResponse(
    id="some_id",
    choices=[
        ChatCompletionResponseChoice(
            index=0,
            message=AI21ChatMessage(role="assistant", content="42"),
        )
    ],
    usage=UsageInfo(
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
    ),
)

_FAKE_STREAM_CHUNKS = [
    ChatCompletionChunk(
        id="some_id_0",
        choices=[
            ChoicesChunk(index=0, delta=ChoiceDelta(role="assistant", content=""))
        ],
    ),
    ChatCompletionChunk(
        id="some_id_1",
        choices=[ChoicesChunk(index=0, delta=ChoiceDelta(role=None, content="42"))],
    ),
]


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in AI21.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_chat():
    messages = ChatMessage(role="user", content="What is the meaning of life?")
    expected_chat_response = ChatResponse(
        message=ChatMessage(role="assistant", content="42"),
        raw=_FAKE_CHAT_COMPLETIONS_RESPONSE.to_dict(),
    )

    with mock.patch("llama_index.llms.ai21.base.AI21Client"):
        llm = AI21(api_key=_FAKE_API_KEY)

    llm._client.chat.completions.create.return_value = _FAKE_CHAT_COMPLETIONS_RESPONSE

    actual_response = llm.chat(messages=[messages])

    assert actual_response == expected_chat_response

    llm._client.chat.completions.create.assert_called_once_with(
        messages=[AI21ChatMessage(role="user", content="What is the meaning of life?")],
        stream=False,
        **llm._get_all_kwargs(),
    )


def test_stream_chat():
    messages = ChatMessage(role="user", content="What is the meaning of life?")

    with mock.patch("llama_index.llms.ai21.base.AI21Client"):
        llm = AI21(api_key=_FAKE_API_KEY)

    llm._client.chat.completions.create.return_value = _FAKE_STREAM_CHUNKS

    actual_response = llm.stream_chat(messages=[messages])

    expected_chunks = [
        ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            delta="",
            raw=_FAKE_STREAM_CHUNKS[0].to_dict(),
        ),
        ChatResponse(
            message=ChatMessage(role="assistant", content="42"),
            delta="42",
            raw=_FAKE_STREAM_CHUNKS[1].to_dict(),
        ),
    ]

    assert list(actual_response) == expected_chunks

    llm._client.chat.completions.create.assert_called_once_with(
        messages=[AI21ChatMessage(role="user", content="What is the meaning of life?")],
        stream=True,
        **llm._get_all_kwargs(),
    )


def test_complete():
    expected_chat_response = CompletionResponse(
        text="42",
        raw=_FAKE_CHAT_COMPLETIONS_RESPONSE.to_dict(),
    )

    with mock.patch("llama_index.llms.ai21.base.AI21Client"):
        llm = AI21(api_key=_FAKE_API_KEY)

    llm._client.chat.completions.create.return_value = _FAKE_CHAT_COMPLETIONS_RESPONSE

    actual_response = llm.complete(prompt="What is the meaning of life?")

    assert actual_response == expected_chat_response

    # Since we actually call chat.completions - check that the call was made to it
    llm._client.chat.completions.create.assert_called_once_with(
        messages=[AI21ChatMessage(role="user", content="What is the meaning of life?")],
        stream=False,
        **llm._get_all_kwargs(),
    )
    llm._client.completion.assert_not_called()


def test_stream_complete():
    expected_stream_completion_chunks_response = [
        CompletionResponse(text="", delta="", raw=_FAKE_STREAM_CHUNKS[0].to_dict()),
        CompletionResponse(text="42", delta="42", raw=_FAKE_STREAM_CHUNKS[1].to_dict()),
    ]

    with mock.patch("llama_index.llms.ai21.base.AI21Client"):
        llm = AI21(api_key=_FAKE_API_KEY)

    llm._client.chat.completions.create.return_value = _FAKE_STREAM_CHUNKS

    actual_response = llm.stream_complete(prompt="What is the meaning of life?")

    assert list(actual_response) == expected_stream_completion_chunks_response

    # Since we actually call chat.completions - check that the call was made to it
    llm._client.chat.completions.create.assert_called_once_with(
        messages=[AI21ChatMessage(role="user", content="What is the meaning of life?")],
        stream=True,
        **llm._get_all_kwargs(),
    )
