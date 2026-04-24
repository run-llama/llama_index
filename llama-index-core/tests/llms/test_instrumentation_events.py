from pydantic import BaseModel

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMCompletionEndEvent,
    LLMCompletionInProgressEvent,
)


class _RawModel(BaseModel):
    answer: str


def test_llm_chat_end_event_does_not_mutate_response_raw() -> None:
    raw = _RawModel(answer="hi")
    response = ChatResponse(message=ChatMessage.from_str("reply"), raw=raw)
    event = LLMChatEndEvent(
        messages=[ChatMessage.from_str("prompt")], response=response
    )

    dumped = event.model_dump()

    assert dumped["response"]["raw"] == {"answer": "hi"}
    assert isinstance(response.raw, _RawModel)
    assert response.raw is raw


def test_llm_chat_in_progress_event_does_not_mutate_response_raw() -> None:
    raw = _RawModel(answer="hi")
    response = ChatResponse(message=ChatMessage.from_str("reply"), raw=raw)
    event = LLMChatInProgressEvent(
        messages=[ChatMessage.from_str("prompt")], response=response
    )

    event.model_dump()

    assert isinstance(response.raw, _RawModel)


def test_llm_completion_end_event_does_not_mutate_response_raw() -> None:
    raw = _RawModel(answer="hi")
    response = CompletionResponse(text="reply", raw=raw)
    event = LLMCompletionEndEvent(prompt="prompt", response=response)

    event.model_dump()

    assert isinstance(response.raw, _RawModel)


def test_llm_completion_in_progress_event_does_not_mutate_response_raw() -> None:
    raw = _RawModel(answer="hi")
    response = CompletionResponse(text="reply", raw=raw)
    event = LLMCompletionInProgressEvent(prompt="prompt", response=response)

    event.model_dump()

    assert isinstance(response.raw, _RawModel)


def test_llm_chat_end_event_handles_none_response() -> None:
    event = LLMChatEndEvent(messages=[ChatMessage.from_str("prompt")], response=None)

    dumped = event.model_dump()

    assert dumped["response"] is None
