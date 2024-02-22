"""Events for AgentSteps."""

from dataclasses import dataclass
from typing import List

from deprecated import deprecated

from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType


@dataclass
class AgentStepStartEventPayload:
    """Payload for AgentStepStartEvent."""

    messages: List[str]


class AgentStepStartEvent(CBEvent):
    """Event to indicate an agent step has started."""

    messages: List[str]

    def __init__(self, messages: List[str]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.AGENT_STEP)
        self.messages = messages

    @property
    @deprecated("You can access the payload properties directly from the class.")
    def payload(self) -> AgentStepStartEventPayload:
        """Return the payload for the event."""
        return AgentStepStartEventPayload(messages=self.messages)


@dataclass
class AgentStepEndEventPayload:
    """Payload for AgentStepEndEvent."""

    response: AgentChatResponse


class AgentStepEndEvent(CBEvent):
    """Event to indicate an agent step has ended."""

    response: AgentChatResponse

    def __init__(self, response: AgentChatResponse):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.AGENT_STEP)
        self.response = response

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> AgentStepEndEventPayload:
        """Return the payload for the event."""
        return AgentStepEndEventPayload(response=self.response)
