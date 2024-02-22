"""Event for the before and after of embedding data into vectors."""

from dataclasses import dataclass
from typing import Any, Dict, List

from deprecated import deprecated

from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType


@dataclass
class EmbeddingStartEventPayload:
    """Payload for EmbeddingStartEvent."""

    serialized: Dict[str, Any]


class EmbeddingStartEvent(CBEvent):
    """Event to indicate texts are being embedded."""

    serialized: Dict[str, Any]

    def __init__(self, serialized: Dict[str, Any]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.EMBEDDING)
        self.serialized = serialized

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> EmbeddingStartEventPayload:
        """Return the payload for the event."""
        return EmbeddingStartEventPayload(serialized=self.serialized)


@dataclass
class EmbeddingEndEventPayload:
    """Payload for EmbeddingEndEvent."""

    chunks: List[str]
    embeddings: List[List[float]]


class EmbeddingEndEvent(CBEvent):
    """Event to indicate texts are done being embedded."""

    chunks: List[str]
    embeddings: List[List[float]]

    def __init__(self, chunks: List[str], embeddings: List[List[float]]):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.EMBEDDING)
        self.chunks = chunks
        self.embeddings = embeddings

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> EmbeddingEndEventPayload:
        """Return the payload for the event."""
        return EmbeddingEndEventPayload(chunks=self.chunks, embeddings=self.embeddings)
