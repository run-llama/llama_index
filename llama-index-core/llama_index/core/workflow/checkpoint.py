import uuid
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
)
from typing import Optional, Dict, Any
from .events import Event


class Checkpoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    last_completed_step: Optional[str]
    input_event: Optional[Event]
    output_event: Event
    ctx_state: Dict[str, Any]
