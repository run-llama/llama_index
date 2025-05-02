from typing import Any
from abc import abstractmethod
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict
from llama_index.core.instrumentation.events.base import BaseEvent


class BaseEventHandler(BaseModel):
    """Base callback handler that can be used to track event starts and ends."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "BaseEventHandler"

    @abstractmethod
    def handle(self, event: BaseEvent, **kwargs: Any) -> Any:
        """Logic for handling event."""
