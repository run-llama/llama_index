"""Base schema for callback managers."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict

TIMESTAMP_FORMAT = "%m/%d/%Y, %H:%M:%S"


class CBEventType(str, Enum):
    """Callback manager event types."""

    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    LLM = "llm"
    # TODO: Can we use these anywhere?
    # QUERY = "query"
    # RETRIEVE = "retrieve"
    # SYNTHESIZE = "synthesize"
    TREE = "tree"


@dataclass
class CBEvent:
    """Generic class to store event information."""

    event_type: CBEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    time: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        """Init time and id if needed."""
        if not self.time:
            self.time = datetime.now().strftime(TIMESTAMP_FORMAT)
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class EventStats:
    """Time-based Statistics for events."""

    total_secs: float
    average_secs: float
    total_count: int
