"""Base schema for callback managers."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class CBEventType(str, Enum):
    """Callback manager event types."""
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    LLM = "llm"
    QUERY = "query"
    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"
    TREE = "tree"
    

@dataclass
class CBEvent:
    event_type: CBEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    time: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        self.time = datetime.now().strftime("%H:%M:%S")
        if not self.id:
            self.id = str(uuid.uuid4())
