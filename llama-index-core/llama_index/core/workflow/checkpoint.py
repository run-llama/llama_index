import uuid
from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import Optional, Dict
from .events import Event


class Checkpoint(BaseModel):
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step: Optional[str]
    event: Event
    ctx_state: Dict
