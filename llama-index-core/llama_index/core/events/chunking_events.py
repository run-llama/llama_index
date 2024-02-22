"""Event for the before and after of text splitting."""

from dataclasses import dataclass
from typing import List, Sequence

from deprecated import deprecated

from llama_index.core.callbacks.schema import CBEvent, CBEventType
from llama_index.core.schema import BaseNode


@dataclass
class ChunkingStartEventPayload:
    """Payload for ChunkingStartEvent."""

    documents: Sequence[BaseNode]


class ChunkingStartEvent(CBEvent):
    """Event to indicate text splitting has begun."""

    documents: Sequence[BaseNode]

    def __init__(self, documents: Sequence[BaseNode]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.CHUNKING)
        self.documents = documents

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> ChunkingStartEventPayload:
        """Return the payload for the event."""
        return ChunkingStartEventPayload(documents=self.documents)


@dataclass
class ChunkingEndEventPayload:
    """Payload for ChunkingEndEvent."""

    text_chunks: List[str]


class ChunkingEndEvent(CBEvent):
    """Event to indicate text splitting has ended."""

    text_chunks: List[str]

    def __init__(self, text_chunks: List[str]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.CHUNKING)
        self.text_chunks = text_chunks

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> ChunkingEndEventPayload:
        """Return the payload for the event."""
        return ChunkingEndEventPayload(text_chunks=self.text_chunks)
