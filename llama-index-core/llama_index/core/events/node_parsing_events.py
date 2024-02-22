"""Event for the before and after documents being split into nodes."""

from dataclasses import dataclass
from typing import List, Sequence

from deprecated import deprecated

from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType
from llama_index.core.schema import BaseNode, Document


@dataclass
class NodeParsingStartEventPayload:
    """Payload for NodeParsingStartEvent."""

    documents: Sequence[Document]


class NodeParsingStartEvent(CBEvent):
    """Event to indicate Documents have begun being parsed into Nodes."""

    documents: Sequence[Document]

    def __init__(self, documents: Sequence[Document]):
        """Initialize the chunking event."""
        super().__init__(event_type=CBEventType.NODE_PARSING)
        self.documents = documents

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> NodeParsingStartEventPayload:
        """Return the payload for the event."""
        return NodeParsingStartEventPayload(documents=self.documents)


@dataclass
class NodeParsingEndEventPayload:
    """Payload for NodeParsingEndEvent."""

    nodes: List[BaseNode]


class NodeParsingEndEvent(CBEvent):
    """Event to indicate Documents have finished being parsed into Nodes."""

    text_chunks: List[str]

    def __init__(self, nodes: List[BaseNode]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.NODE_PARSING)
        self.nodes = nodes

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> NodeParsingEndEventPayload:
        """Return the payload for the event."""
        return NodeParsingEndEventPayload(nodes=self.nodes)
