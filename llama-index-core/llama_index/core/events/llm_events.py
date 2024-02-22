"""Event for the before and after interacting with an LLM."""

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from deprecated import deprecated

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType


@dataclass
class LLMStartEventPayload:
    """Payload for LLMStartEvent."""

    messages: Sequence[ChatMessage]
    additional_kwargs: Dict[str, Any]
    serialized: Dict[str, Any]


class LLMStartEvent(CBEvent):
    """Event to indicate we begin reaching out to an LLM."""

    messages: Sequence[ChatMessage]
    additional_kwargs: Dict[str, Any]
    serialized: Dict[str, Any]

    def __init__(
        self,
        messages: Sequence[ChatMessage],
        additional_kwargs: Dict[str, Any],
        serialized: Dict[str, Any],
    ):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.LLM)
        self.messages = messages
        self.additional_kwargs = additional_kwargs
        self.serialized = serialized

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> LLMStartEventPayload:
        """Return the payload for the event."""
        return LLMStartEventPayload(
            messages=self.messages,
            additional_kwargs=self.additional_kwargs,
            serialized=self.serialized,
        )


@dataclass
class LLMEndEventPayload:
    """Payload for LLMEndEvent."""

    messages: Sequence[ChatMessage]
    last_response: ChatMessage


class LLMEndEvent(CBEvent):
    """Event to indicate the LLM response has completed."""

    messages: Sequence[ChatMessage]
    last_response: ChatMessage

    def __init__(self, messages: Sequence[ChatMessage], last_response: ChatMessage):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.LLM)
        self.messages = messages
        self.last_response = last_response

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> LLMEndEventPayload:
        """Return the payload for the event."""
        return LLMEndEventPayload(
            messages=self.messages, last_response=self.last_response
        )
