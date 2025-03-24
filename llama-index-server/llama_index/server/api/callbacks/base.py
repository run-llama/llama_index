import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("uvicorn")


class EventCallback(ABC):
    """
    Base class for event callbacks during event streaming.
    """

    async def run(self, event: Any) -> Any:
        """
        Called for each event in the stream.
        Default behavior: pass through the event unchanged.
        """
        return event

    async def on_complete(self, final_response: str) -> Any:
        """
        Called when the stream is complete.
        Default behavior: return None.
        """
        return None

    @abstractmethod
    def from_default(self, *args, **kwargs) -> "EventCallback":
        """
        Create a new instance of the processor from default values.
        """
