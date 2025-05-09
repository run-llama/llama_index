from __future__ import annotations

import json
from typing import Type, Any, Iterable
from unittest import mock

import pytest
from ai21.models import (
    CompletionsResponse as J2CompletionResponse,
    CompletionData,
    Completion,
    CompletionFinishReason,
    Prompt,
    RoleType,
    ChatMessage as J2ChatMessage,
    ChatResponse as J2ChatResponse,
    ChatOutput,
)
from ai21.models.chat import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionChunk,
    ChoicesChunk,
    ChoiceDelta,
    ChatMessage as AI21ChatMessage,
    AssistantMessage,
    ToolCall,
    ToolFunction,
    ToolMessage as AI21ToolMessage,
    SystemMessage,
    UserMessage,
)
from ai21.models.usage_info import UsageInfo
from ai21_tokenizer import JurassicTokenizer, JambaInstructTokenizer, BaseTokenizer
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.llms import ChatMessage

from llama_index.llms.ai21 import AI21
from llama_index.llms.ai21.utils import (
    from_ai21_message_to_chat_message,
    is_function_calling_model,
    message_to_ai21_message,
)

_PROMPT = "What is the meaning of life?"
_FAKE_API_KEY = "fake-api-key"
_MODEL_NAME = "jamba-instruct"
_FAKE_CHAT_COMPLETIONS_RESPONSE = ChatCompletionResponse(
    id="some_id",
    choices=[
        ChatCompletionResponseChoice(
            index=0,
            message=AssistantMessage(role="assistant", content="42"),
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

_FAKE_RAW_COMPLETION_RESPONSE = {
    "id": "1",
    "prompt": {"text": "What is the meaning of life?", "tokens": None},
    "completions": [
        {
            "data": {"text": "42", "tokens": None},
            "finishReason": {
                "reason": None,
                "length": None,
            },
        }
    ],
}

_FAKE_COMPLETION_RESPONSE = CompletionResponse(
    text="42",
    raw=_FAKE_RAW_COMPLETION_RESPONSE,
)


class AsyncIterator:
    def __init__(self, iterable: Iterable) -> None:
        self._iterable = iter(iterable)

    def __aiter__(self) -> AsyncIterator:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iterable)
        except StopIteration:
            raise StopAsyncIteration


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in AI21.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_chat():
    messages = ChatMessage(role="user", content="What is the meaning of life?")
    expected_chat_response = ChatResponse(
        message=ChatMessage(role="assistant", content="42"),
        raw=_FAKE_CHAT_COMPLETIONS_RESPONSE.model_dump(),
    )

    with mock.patch("llama_index.llms.ai21.base.AI21Client"):
        llm = AI21(api_key=_FAKE_API_KEY)

    llm._client.chat.completions.create.return_value = _FAKE_CHAT_COMPLETIONS_RESPONSE

    actual_response = llm.chat(messages=[messages])

    assert actual_response == expected_chat_response

    llm._client.chat.completions.create.assert_called_once_with(
        messages=[UserMessage(content="What is the meaning of life?")],
        stream=False,
        **llm._get_all_kwargs(),
    )


def test_chat__when_j2():
    j2_response = J2ChatResponse(
        outputs=[ChatOutput(text="42", role=RoleType.ASSISTANT, finish_reason=None)],
    )
    expected_chat_response = ChatResponse(
        message=ChatMessage(role="assistant", content="42"),
        raw=j2_response.to_dict(),
    )

    with mock.patch("llama_index.llms.ai21.base.AI21Client"):
        llm = AI21(api_key=_FAKE_API_KEY, model="j2-ultra")

    llm._client.chat.create.return_value = j2_response

    actual_response = llm.chat(messages=[ChatMessage(role="user", content=_PROMPT)])

    assert actual_response == expected_chat_response

    llm._client.chat.create.assert_called_once_with(
        system="",
        messages=[J2ChatMessage(role=RoleType.USER, text=_PROMPT)],
        stream=False,
        **llm._get_all_kwargs(),
    )


def test_chat__when_j2_and_system_message_not_first__should_raise_error():
    llm = AI21(api_key=_FAKE_API_KEY, model="j2-ultra")

    with pytest.raises(ValueError) as e:
        llm.chat(
            messages=[
                ChatMessage(role="user", content=_PROMPT),
                ChatMessage(role="system", content=""),
            ]
        )

    assert e.value.args[0] == "System message must be at beginning of message list."


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
        messages=[UserMessage(content="What is the meaning of life?")],
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
        messages=[UserMessage(content="What is the meaning of life?")],
        stream=False,
        **llm._get_all_kwargs(),
    )
    llm._client.completion.assert_not_called()


def test_complete__when_j2():
    prompt = "What is the meaning of life?"

    with mock.patch("llama_index.llms.ai21.base.AI21Client"):
        llm = AI21(api_key=_FAKE_API_KEY, model="j2-ultra")

    expected_response = J2CompletionResponse(
        id="1",
        completions=[
            Completion(
                data=CompletionData(text="42"), finish_reason=CompletionFinishReason()
            )
        ],
        prompt=Prompt(text=prompt),
    )

    llm._client.completion.create.return_value = expected_response

    actual_response = llm.complete(prompt=prompt)

    assert actual_response == _FAKE_COMPLETION_RESPONSE

    # Since we actually call chat.completions - check that the call was made to it
    llm._client.completion.create.assert_called_once_with(
        prompt=prompt,
        stream=False,
        **llm._get_all_kwargs(),
    )


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
        messages=[UserMessage(content="What is the meaning of life?")],
        stream=True,
        **llm._get_all_kwargs(),
    )


@pytest.mark.asyncio
async def test_achat():
    messages = ChatMessage(role="user", content="What is the meaning of life?")
    expected_chat_response = ChatResponse(
        message=ChatMessage(role="assistant", content="42"),
        raw=_FAKE_CHAT_COMPLETIONS_RESPONSE.to_dict(),
    )

    with mock.patch(
        "llama_index.llms.ai21.base.AsyncAI21Client", side_effect=mock.AsyncMock
    ):
        llm = AI21(api_key=_FAKE_API_KEY)

    llm._async_client.chat.completions.create.return_value = (
        _FAKE_CHAT_COMPLETIONS_RESPONSE
    )

    actual_response = await llm.achat(messages=[messages])

    assert actual_response == expected_chat_response

    llm._async_client.chat.completions.create.assert_called_once_with(
        messages=[UserMessage(content="What is the meaning of life?")],
        stream=False,
        **llm._get_all_kwargs(),
    )


@pytest.mark.asyncio
async def test_acomplete():
    expected_chat_response = CompletionResponse(
        text="42",
        raw=_FAKE_CHAT_COMPLETIONS_RESPONSE.to_dict(),
    )

    with mock.patch(
        "llama_index.llms.ai21.base.AsyncAI21Client", side_effect=mock.AsyncMock
    ):
        llm = AI21(api_key=_FAKE_API_KEY)

    llm._async_client.chat.completions.create.return_value = (
        _FAKE_CHAT_COMPLETIONS_RESPONSE
    )

    actual_response = await llm.acomplete(prompt="What is the meaning of life?")

    assert actual_response == expected_chat_response

    # Since we actually call chat.completions - check that the call was made to it
    llm._async_client.chat.completions.create.assert_called_once_with(
        messages=[UserMessage(content="What is the meaning of life?")],
        stream=False,
        **llm._get_all_kwargs(),
    )
    llm._async_client.completion.assert_not_called()


@pytest.mark.asyncio
async def test_astream_chat():
    messages = ChatMessage(role="user", content="What is the meaning of life?")

    with mock.patch(
        "llama_index.llms.ai21.base.AsyncAI21Client", side_effect=mock.AsyncMock
    ):
        llm = AI21(api_key=_FAKE_API_KEY)

    llm._async_client.chat.completions.create.return_value = AsyncIterator(
        _FAKE_STREAM_CHUNKS
    )

    actual_response = await llm.astream_chat(messages=[messages])

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

    actual_response = [r async for r in actual_response]

    assert list(actual_response) == expected_chunks

    llm._async_client.chat.completions.create.assert_called_once_with(
        messages=[UserMessage(content="What is the meaning of life?")],
        stream=True,
        **llm._get_all_kwargs(),
    )


@pytest.mark.asyncio
async def test_astream_complete():
    expected_stream_completion_chunks_response = [
        CompletionResponse(text="", delta="", raw=_FAKE_STREAM_CHUNKS[0].to_dict()),
        CompletionResponse(text="42", delta="42", raw=_FAKE_STREAM_CHUNKS[1].to_dict()),
    ]

    with mock.patch(
        "llama_index.llms.ai21.base.AsyncAI21Client", side_effect=mock.AsyncMock
    ):
        llm = AI21(api_key=_FAKE_API_KEY)
        llm._async_client.chat.completions.create.return_value = AsyncIterator(
            _FAKE_STREAM_CHUNKS
        )

    actual_response = await llm.astream_complete(prompt="What is the meaning of life?")

    actual_response = [r async for r in actual_response]

    assert actual_response == expected_stream_completion_chunks_response

    # Since we actually call chat.completions - check that the call was made to it
    llm._async_client.chat.completions.create.assert_called_once_with(
        messages=[UserMessage(content="What is the meaning of life?")],
        stream=True,
        **llm._get_all_kwargs(),
    )


def test_stream_complete_when_j2__should_raise_error():
    llm = AI21(api_key=_FAKE_API_KEY, model="j2-ultra")

    with pytest.raises(ValueError):
        llm.stream_complete(prompt="What is the meaning of life?")


def test_chat_complete_when_j2__should_raise_error():
    llm = AI21(api_key=_FAKE_API_KEY, model="j2-ultra")

    with pytest.raises(ValueError):
        llm.stream_chat(
            messages=[ChatMessage(role="user", content="What is the meaning of life?")]
        )


@pytest.mark.asyncio
async def test_achat_complete_when_j2__should_raise_error():
    llm = AI21(api_key=_FAKE_API_KEY, model="j2-ultra")

    with pytest.raises(ValueError):
        await llm.astream_chat(
            messages=[ChatMessage(role="user", content="What is the meaning of life?")]
        )


@pytest.mark.asyncio
async def test_astream_complete_when_j2__should_raise_error():
    llm = AI21(api_key=_FAKE_API_KEY, model="j2-ultra")

    with pytest.raises(ValueError):
        await llm.astream_complete(prompt="What is the meaning of life?")


@pytest.mark.parametrize(
    ids=[
        "when_j2_mid__should_return_right_tokenizer",
        "when_j2_ultra__should_return_right_tokenizer",
        "when_jamba_instruct__should_return_right_tokenizer",
    ],
    argnames=("model", "expected_tokenizer_type"),
    argvalues=[
        ("j2-mid", JurassicTokenizer),
        ("j2-ultra", JurassicTokenizer),
        ("jamba-instruct", JambaInstructTokenizer),
    ],
)
def test_tokenizer(model: str, expected_tokenizer_type: Type[BaseTokenizer]):
    llm = AI21(api_key=_FAKE_API_KEY, model=model)
    assert isinstance(llm.tokenizer, expected_tokenizer_type)


def test_from_ai21_message_to_chat_message_no_tool_calls():
    ai21_message = AssistantMessage(role="assistant", content="Hello!", tool_calls=None)

    chat_message = from_ai21_message_to_chat_message(ai21_message)

    assert isinstance(chat_message, ChatMessage)
    assert chat_message.role == "assistant"
    assert chat_message.content == "Hello!"
    assert chat_message.additional_kwargs == {}


def test_from_ai21_message_to_chat_message_with_tool_calls():
    tool_call = ToolCall(
        id="some_id",
        function=ToolFunction(
            name="some_function",
            arguments=json.dumps({"x": 42}),
        ),
    )
    ai21_message = AssistantMessage(
        role="assistant", content="Here is a response.", tool_calls=[tool_call]
    )

    chat_message = from_ai21_message_to_chat_message(ai21_message)

    assert isinstance(chat_message, ChatMessage)
    assert chat_message.role == "assistant"
    assert chat_message.content == "Here is a response."
    assert chat_message.additional_kwargs == {"tool_calls": [tool_call]}


@pytest.mark.parametrize(
    ids=[
        "when_j2_mid__should_return_false",
        "when_j2_ultra__should_return_false",
        "when_jamba-instruct__should_return_false",
        "when_jamba-1.5-mini__should_return_true",
        "when_jamba-1.5-large__should_return_true",
    ],
    argnames=("model", "expected_result"),
    argvalues=[
        ("j2-mid", False),
        ("j2-ultra", False),
        ("jamba-instruct", False),
        ("jamba-1.5-mini", True),
        ("jamba-1.5-large", True),
    ],
)
def test_is_function_calling_model(model: str, expected_result: bool):
    assert is_function_calling_model(model) == expected_result


@pytest.mark.parametrize(
    ids=[
        "when_tool_message",
        "when_user_message",
        "when_assistant_message",
        "when_system_message",
    ],
    argvalues=[
        (
            ChatMessage(
                role=MessageRole.TOOL,
                content="Tool message",
                additional_kwargs={"tool_call_id": "tool_id_1"},
            ),
            AI21ToolMessage(content="Tool message", tool_call_id="tool_id_1"),
        ),
        (
            ChatMessage(role=MessageRole.USER, content="User message"),
            UserMessage(content="User message"),
        ),
        (
            ChatMessage(role=MessageRole.ASSISTANT, content="Assistant message"),
            AssistantMessage(content="Assistant message"),
        ),
        (
            ChatMessage(role=MessageRole.SYSTEM, content="Assistant message"),
            SystemMessage(content="Assistant message"),
        ),
    ],
    argnames=("message", "expected_result"),
)
def test_message_to_ai21_message(
    message: ChatMessage, expected_result: AI21ChatMessage
) -> None:
    result = message_to_ai21_message(message)
    assert result == expected_result
