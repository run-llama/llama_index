"""Event for the before and after response synthesis."""

from dataclasses import dataclass

from deprecated import deprecated

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.schema import CBEvent, CBEventType


@dataclass
class SynthesizeStartEventPayload:
    """Payload for SynthesizeStartEvent."""

    query_str: str


class SynthesizeStartEvent(CBEvent):
    """Event to indicate a response has started being synthesized."""

    query_str: str

    def __init__(self, query_str: str):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.SYNTHESIZE)
        self.query_str = query_str

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> SynthesizeStartEventPayload:
        """Return the payload for the event."""
        return SynthesizeStartEventPayload(query_str=self.query_str)


@dataclass
class SynthesizeEndEventPayload:
    """Payload for SynthesizeEndEvent."""

    response: RESPONSE_TYPE


class SynthesizeEndEvent(CBEvent):
    """Event to indicate a response has finished being synthesized."""

    response: RESPONSE_TYPE

    def __init__(self, response: RESPONSE_TYPE):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.SYNTHESIZE)
        self.response = response

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> SynthesizeEndEventPayload:
        """Return the payload for the event."""
        return SynthesizeEndEventPayload(response=self.response)
