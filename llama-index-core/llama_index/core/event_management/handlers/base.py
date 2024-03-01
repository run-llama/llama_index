import logging
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any

from llama_index.core.callbacks.schema import BASE_TRACE_EVENT

logger = logging.getLogger(__name__)
global_stack_trace = ContextVar("trace", default=[BASE_TRACE_EVENT])


class BaseEventHandler(ABC):
    """Base callback handler that can be used to track event starts and ends."""

    @abstractmethod
    @classmethod
    def class_name(cls) -> str:
        """Class name."""

    @abstractmethod
    def handle(event: Any) -> None:
        """Logic for handling event."""
