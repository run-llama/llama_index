"""Event for the before and after a Query operation."""

from dataclasses import dataclass

from deprecated import deprecated

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType


@dataclass
class QueryStartEventPayload:
    """Payload for QueryStartEvent."""

    query_str: str


class QueryStartEvent(CBEvent):
    """Event to indicate texts are being embedded."""

    query_str: str

    def __init__(self, query_str: str):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.QUERY)
        self.query_str = query_str

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> QueryStartEventPayload:
        """Return the payload for the event."""
        return QueryStartEventPayload(query_str=self.query_str)


@dataclass
class QueryEndEventPayload:
    """Payload for QueryEndEvent."""

    response: RESPONSE_TYPE


class QueryEndEvent(CBEvent):
    """Event to indicate texts are done being embedded."""

    response: RESPONSE_TYPE

    def __init__(self, response: RESPONSE_TYPE):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.QUERY)
        self.response = response

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> QueryEndEventPayload:
        """Return the payload for the event."""
        return QueryEndEventPayload(response=self.response)
