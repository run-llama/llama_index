from typing import Any

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMCompletionEndEvent,
    LLMCompletionInProgressEvent,
)


class _RawModel(BaseModel):
    answer: str


class _ChatEventWrapper(BaseModel):
    event: LLMChatEndEvent


class _CompletionEventWrapper(BaseModel):
    event: LLMCompletionEndEvent


def test_chat_response_serializes_raw_model_without_mutating() -> None:
    raw = _RawModel(answer="hello")
    response = ChatResponse(message=ChatMessage.from_str("reply"), raw=raw)

    dumped = response.model_dump()

    assert dumped["raw"] == {"answer": "hello"}
    assert response.raw is raw
    assert isinstance(response.raw, _RawModel)


def test_completion_response_serializes_raw_model_without_mutating() -> None:
    raw = _RawModel(answer="hello")
    response = CompletionResponse(text="reply", raw=raw)

    dumped = response.model_dump()

    assert dumped["raw"] == {"answer": "hello"}
    assert response.raw is raw
    assert isinstance(response.raw, _RawModel)


@pytest.mark.parametrize(
    ("event", "response"),
    [
        (
            LLMChatInProgressEvent(
                messages=[ChatMessage.from_str("prompt")],
                response=ChatResponse(
                    message=ChatMessage.from_str("reply"),
                    raw=_RawModel(answer="chat in progress"),
                ),
            ),
            "chat in progress",
        ),
        (
            LLMChatEndEvent(
                messages=[ChatMessage.from_str("prompt")],
                response=ChatResponse(
                    message=ChatMessage.from_str("reply"),
                    raw=_RawModel(answer="chat end"),
                ),
            ),
            "chat end",
        ),
        (
            LLMCompletionInProgressEvent(
                prompt="prompt",
                response=CompletionResponse(
                    text="reply",
                    raw=_RawModel(answer="completion in progress"),
                ),
            ),
            "completion in progress",
        ),
        (
            LLMCompletionEndEvent(
                prompt="prompt",
                response=CompletionResponse(
                    text="reply",
                    raw=_RawModel(answer="completion end"),
                ),
            ),
            "completion end",
        ),
    ],
)
def test_llm_events_serialize_raw_model_without_mutating(
    event: Any, response: str
) -> None:
    raw = event.response.raw

    dumped = event.model_dump()

    assert dumped["response"]["raw"] == {"answer": response}
    assert event.response.raw is raw
    assert isinstance(event.response.raw, _RawModel)


def test_nested_chat_event_serializes_raw_model() -> None:
    raw = _RawModel(answer="nested chat")
    event = LLMChatEndEvent(
        messages=[ChatMessage.from_str("prompt")],
        response=ChatResponse(message=ChatMessage.from_str("reply"), raw=raw),
    )

    dumped = _ChatEventWrapper(event=event).model_dump()

    assert dumped["event"]["response"]["raw"] == {"answer": "nested chat"}
    assert event.response is not None
    assert event.response.raw is raw


def test_nested_completion_event_serializes_raw_model() -> None:
    raw = _RawModel(answer="nested completion")
    event = LLMCompletionEndEvent(
        prompt="prompt",
        response=CompletionResponse(text="reply", raw=raw),
    )

    dumped = _CompletionEventWrapper(event=event).model_dump()

    assert dumped["event"]["response"]["raw"] == {"answer": "nested completion"}
    assert event.response.raw is raw
