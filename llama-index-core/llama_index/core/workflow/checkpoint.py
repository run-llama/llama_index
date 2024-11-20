import uuid
from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import Optional, Dict
from .events import Event


class Checkpoint(BaseModel):
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    last_completed_step: Optional[str]
    input_event: Optional[Event]
    output_event: Event
    ctx_state: Dict
