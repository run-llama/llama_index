"""A CallbackHandler registers with a CallbackManager and receives events from the manager."""

import logging
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from deprecated import deprecated

from llama_index.core.callbacks.schema import BASE_TRACE_EVENT, CBEvent, CBEventType

logger = logging.getLogger(__name__)
global_stack_trace = ContextVar("trace", default=[BASE_TRACE_EVENT])


class BaseCallbackHandler(ABC):
    """Base callback handler that can be used to track event starts and ends."""

    _events_to_ignore: tuple[CBEventType, ...]

    def __init__(
        self,
        event_starts_to_ignore: List[CBEventType],
        event_ends_to_ignore: List[CBEventType],
    ) -> None:
        """Initialize the base callback handler."""
        self._events_to_ignore = tuple(event_starts_to_ignore + event_ends_to_ignore)

    @property
    def event_starts_to_ignore(self) -> tuple[CBEventType, ...]:
        """Return the events to ignore."""
        return self._events_to_ignore

    @property
    def event_ends_to_ignore(self) -> tuple[CBEventType, ...]:
        """Return the events to ignore."""
        return self._events_to_ignore

    @abstractmethod
    @deprecated("Events no longer have starts or ends. You should call on_event now")
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
    @deprecated("Events no longer have starts or ends. You should call on_event now")
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""

    @abstractmethod
    def on_event(
        self,
        event: CBEvent,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an event is fired."""

    @abstractmethod
    @deprecated("Starting traces should happen in the manager, not the event")
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""

    @abstractmethod
    @deprecated("Ending traces should happen in the manager, not the event")
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
