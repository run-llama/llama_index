"""Base schema for callback managers."""
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict


class CBEventType(str, Enum):
    """Callback manager event types."""
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    LLM = "llm"
    QUERY = "query"
    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"
    

@dataclass
class CBEvent:
    event_type: CBEventType
    payload: Dict[str, str]
    time: str = ""
    id: str = ""

    def __post_init__(self):
        self.time = datetime.now().strftime("%H:%M:%S")
        if not self.id:
            self.id = str(uuid.uuid4())
