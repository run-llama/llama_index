import logging
from abc import abstractmethod
from contextvars import ContextVar

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.schema import BASE_TRACE_EVENT
from llama_index.core.instrumentation.events.base import BaseEvent

logger = logging.getLogger(__name__)
global_stack_trace = ContextVar("trace", default=[BASE_TRACE_EVENT])


class BaseEventHandler(BaseModel):
    """Base callback handler that can be used to track event starts and ends."""

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "BaseEventHandler"

    @abstractmethod
    def handle(self, event: BaseEvent) -> None:
        """Logic for handling event."""

    @abstractmethod
    def span_enter(self, id: str) -> None:
        """Logic for entering a span."""

    @abstractmethod
    def span_exit(self, id: str) -> None:
        """Logic for exiting a span."""

    class Config:
        arbitrary_types_allowed = True
