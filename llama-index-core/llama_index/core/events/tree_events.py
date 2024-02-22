"""Events for trees."""

from dataclasses import dataclass
from typing import List

from deprecated import deprecated

from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType


@dataclass
class TreeStartEventPayload:
    """Payload for TreeStartEvent."""

    chunks: List[str]


class TreeStartEvent(CBEvent):
    """Event to indicate a tree has started."""

    chunks: List[str]

    def __init__(self, chunks: List[str]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.TREE)
        self.chunks = chunks

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> TreeStartEventPayload:
        """Return the payload for the event."""
        return TreeStartEventPayload(chunks=self.chunks)


@dataclass
class TreeEndEventPayload:
    """Payload for TreeEndEvent."""

    summaries: List[str]
    level: int


class TreeEndEvent(CBEvent):
    """Event to indicate a tree has ended."""

    summaries: List[str]
    level: int

    def __init__(self, summaries: List[str], level: int):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.TREE)
        self.summaries = summaries
        self.level = level

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> TreeEndEventPayload:
        """Return the payload for the event."""
        return TreeEndEventPayload(summaries=self.summaries, level=self.level)
