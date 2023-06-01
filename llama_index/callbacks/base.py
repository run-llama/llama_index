import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Generator

from llama_index.callbacks.schema import CBEventType, LEAF_EVENTS, BASE_TRACE_ID


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
        **kwargs: Any
    ) -> str:
        """Run when an event starts and return id of event."""

    @abstractmethod
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
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


class CallbackManager(BaseCallbackHandler, ABC):
    """Callback manager that handles callbacks for events within LlamaIndex.

    The callback manager provides a way to call handlers on event starts/ends.

    Additionally, the callback manager traces the current stack of events.
    It does this by using a few key attributes.
    - trace_stack - The current stack of events that have not ended yet.
                    When an event ends, it's remove from the stack.
    - trace_map - A mapping of event ids to their children events.
                  On the start of events, the bottom of the trace stack
                  is used as the current parent event for the trace map.
    - trace_id - A simple name for the current trace, usually denoting the
                 entrypoint (query, index_construction, insert, etc.)

    Args:
        handlers (List[BaseCallbackHandler]): list of handlers to use.

    """

    def __init__(self, handlers: List[BaseCallbackHandler]):
        """Initialize the manager with a list of handlers."""
        self.handlers = handlers
        self._trace_map: Dict[str, List[str]] = defaultdict(list)
        self._trace_stack: List[str] = [BASE_TRACE_ID]
        self._trace_id: Optional[str] = None

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> str:
        """Run handlers when an event starts and return id of event."""
        event_id = event_id or str(uuid.uuid4())
        self._trace_map[self._trace_stack[-1]].append(event_id)
        for handler in self.handlers:
            if event_type not in handler.event_starts_to_ignore:
                handler.on_event_start(event_type, payload, event_id=event_id, **kwargs)

        if event_type not in LEAF_EVENTS:
            self._trace_stack.append(event_id)

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> None:
        """Run handlers when an event ends."""
        event_id = event_id or str(uuid.uuid4())
        for handler in self.handlers:
            if event_type not in handler.event_ends_to_ignore:
                handler.on_event_end(event_type, payload, event_id=event_id, **kwargs)

        if event_type not in LEAF_EVENTS:
            self._trace_stack.pop()

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = handlers

    @contextmanager
    def as_trace(self, trace_id: str) -> Generator[None, None, None]:
        """Context manager tracer for lanching and shutdown of traces."""
        self.start_trace(trace_id=trace_id)
        yield
        self.end_trace(trace_id=trace_id)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        if not self._trace_id:
            self._reset_trace_events()

            for handler in self.handlers:
                handler.start_trace(trace_id=trace_id)

            self._trace_id = trace_id

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        if trace_id is not None and trace_id == self._trace_id:
            for handler in self.handlers:
                handler.end_trace(trace_id=trace_id, trace_map=self._trace_map)
            self._trace_id = None

    def _reset_trace_events(self) -> None:
        """Helper function to reset the current trace."""
        self._trace_map = defaultdict(list)
        self._trace_stack = [BASE_TRACE_ID]

    @property
    def trace_map(self) -> Dict[str, List[str]]:
        return self._trace_map
