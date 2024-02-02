import logging
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from llama_index.callbacks.schema import BASE_TRACE_EVENT, CBEventType

logger = logging.getLogger(__name__)
global_stack_trace = ContextVar("trace", default=[BASE_TRACE_EVENT])


class BaseCallbackHandler(ABC):
    """Base callback handler that can be used to track event starts and ends."""

    def __init__(
        self,
        event_starts_to_ignore: List[CBEventType],
        event_ends_to_ignore: List[CBEventType],
    ) -> None:
        """Initialize the base callback handler."""
        self.event_starts_to_ignore = tuple(event_starts_to_ignore)
        self.event_ends_to_ignore = tuple(event_ends_to_ignore)

    @abstractmethod
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""

    @abstractmethod
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""

    @abstractmethod
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""

    @abstractmethod
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
