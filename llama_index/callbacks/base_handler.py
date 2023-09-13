import logging
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.callbacks.schema import CBEventType, BASE_TRACE_EVENT

logger = logging.getLogger(__name__)
global_stack_trace = ContextVar("trace", default=[BASE_TRACE_EVENT])


class BaseCallbackHandler(BaseModel, ABC):
    """Base callback handler that can be used to track event starts and ends."""

    event_starts_to_ignore: List[CBEventType] = Field(
        description="List of event types to ignore on start.", default_factory=list
    )
    event_ends_to_ignore: List[CBEventType] = Field(
        description="List of event types to ignore on end.", default_factory=list
    )

    @abstractmethod
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
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
