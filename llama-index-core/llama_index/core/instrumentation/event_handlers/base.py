from typing import Any
from abc import abstractmethod
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.instrumentation.events.base import BaseEvent


class BaseEventHandler(BaseModel):
    """Base callback handler that can be used to track event starts and ends."""

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "BaseEventHandler"

    @abstractmethod
    def handle(self, event: BaseEvent, **kwargs) -> Any:
        """Logic for handling event."""

    class Config:
        arbitrary_types_allowed = True
