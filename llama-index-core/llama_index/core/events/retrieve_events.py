"""Event for the before and after retrieval of nodes."""

from dataclasses import dataclass
from typing import List

from deprecated import deprecated

from llama_index.core.callbacks.schema import CBEvent, CBEventType
from llama_index.core.schema import NodeWithScore


@dataclass
class RetrieveStartEventPayload:
    """Payload for RetrieveStartEvent."""

    query_str: str


class RetrieveStartEvent(CBEvent):
    """Event to indicate nodes are being retrieved."""

    query_str: str

    def __init__(self, query_str: str):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.RETRIEVE)
        self.query_str = query_str

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> RetrieveStartEventPayload:
        """Return the payload for the event."""
        return RetrieveStartEventPayload(query_str=self.query_str)


@dataclass
class RetrieveEndEventPayload:
    """Payload for RetrieveEndEvent."""

    nodes: List[NodeWithScore]


class RetrieveEndEvent(CBEvent):
    """Event to indicate nodes have been retrieved."""

    nodes: List[NodeWithScore]

    def __init__(self, nodes: List[NodeWithScore]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.RETRIEVE)
        self.nodes = nodes

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> RetrieveEndEventPayload:
        """Return the payload for the event."""
        return RetrieveEndEventPayload(nodes=self.nodes)
