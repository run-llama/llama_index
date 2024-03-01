from llama_index.core.bridge.pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime


class BaseEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    id_: str = Field(default_factory=lambda: uuid4())

    @classmethod
    def class_name(cls):
        """Return class name."""
        return "BaseEvent"

    class Config:
        arbitrary_types_allowed = True
