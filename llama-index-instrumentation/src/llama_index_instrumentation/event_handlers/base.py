from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from llama_index_instrumentation.base import BaseEvent


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

    async def ahandle(self, event: BaseEvent, **kwargs: Any) -> Any:
        return self.handle(event, **kwargs)
