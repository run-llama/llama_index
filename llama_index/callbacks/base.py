import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Generator

from llama_index.callbacks.schema import CBEventType, LEAF_EVENTS, BASE_TRACE_EVENT


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


class CallbackManager(BaseCallbackHandler, ABC):
    """
    Callback manager that handles callbacks for events within LlamaIndex.

    The callback manager provides a way to call handlers on event starts/ends.

    Additionally, the callback manager traces the current stack of events.
    It does this by using a few key attributes.
    - trace_stack - The current stack of events that have not ended yet.
                    When an event ends, it's removed from the stack.
                    Since this is a contextvar, it is unique to each
                    thread/task.
    - trace_map - A mapping of event ids to their children events.
                  On the start of events, the bottom of the trace stack
                  is used as the current parent event for the trace map.
    - trace_id - A simple name for the current trace, usually denoting the
                 entrypoint (query, index_construction, insert, etc.)

    Args:
        handlers (List[BaseCallbackHandler]): list of handlers to use.

    Usage:
        with callback_manager.event(CBEventType.QUERY) as event:
            event.on_start(payload={key, val})
            ...
            event.on_end(payload={key, val})

    """

    def __init__(self, handlers: List[BaseCallbackHandler]):
        """Initialize the manager with a list of handlers."""
        self.handlers = handlers
        self._trace_map: Dict[str, List[str]] = defaultdict(list)
        self._trace_event_stack: ContextVar[List[str]] = ContextVar(
            "trace", default=[BASE_TRACE_EVENT]
        )
        self._trace_id_stack: List[str] = []

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Run handlers when an event starts and return id of event."""
        event_id = event_id or str(uuid.uuid4())

        parent_id = self._trace_event_stack.get()[-1]
        self._trace_map[parent_id].append(event_id)
        for handler in self.handlers:
            if event_type not in handler.event_starts_to_ignore:
                handler.on_event_start(event_type, payload, event_id=event_id, **kwargs)

        if event_type not in LEAF_EVENTS:
            current_trace_stack = self._trace_event_stack.get().copy()
            current_trace_stack.append(event_id)
            self._trace_event_stack.set(current_trace_stack)

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run handlers when an event ends."""
        event_id = event_id or str(uuid.uuid4())
        for handler in self.handlers:
            if event_type not in handler.event_ends_to_ignore:
                handler.on_event_end(event_type, payload, event_id=event_id, **kwargs)

        if event_type not in LEAF_EVENTS:
            current_trace_stack = self._trace_event_stack.get().copy()
            current_trace_stack.pop()
            self._trace_event_stack.set(current_trace_stack)

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
    def event(self, event_type: CBEventType) -> Generator["EventContext", None, None]:
        yield EventContext(self, event_type)

    @contextmanager
    def as_trace(self, trace_id: str) -> Generator[None, None, None]:
        """Context manager tracer for lanching and shutdown of traces."""
        self.start_trace(trace_id=trace_id)
        yield
        self.end_trace(trace_id=trace_id)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        if trace_id is not None:
            if len(self._trace_id_stack) == 0:
                self._reset_trace_events()

                for handler in self.handlers:
                    handler.start_trace(trace_id=trace_id)

                self._trace_id_stack = [trace_id]
            else:
                self._trace_id_stack.append(trace_id)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        if trace_id is not None and len(self._trace_id_stack) > 0:
            self._trace_id_stack.pop()
            if len(self._trace_id_stack) == 0:
                for handler in self.handlers:
                    handler.end_trace(trace_id=trace_id, trace_map=self._trace_map)
                self._trace_id_stack = []

    def _reset_trace_events(self) -> None:
        """Helper function to reset the current trace."""

        self._trace_map = defaultdict(list)
        self._trace_event_stack.set([BASE_TRACE_EVENT])

    @property
    def trace_map(self) -> Dict[str, List[str]]:
        return self._trace_map


class EventContext:
    """
    Simple wrapper to call callbacks on event starts and ends
    with an event type and id.
    """

    def __init__(self, callback_manager: CallbackManager, event_type: CBEventType):
        self._callback_manager = callback_manager
        self._event_type = event_type
        self._event_id: Optional[str] = None

    def on_start(self, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._event_id = self._callback_manager.on_event_start(
            self._event_type, payload=payload, event_id=self._event_id, **kwargs
        )

    def on_end(self, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._callback_manager.on_event_end(
            self._event_type, payload=payload, event_id=self._event_id, **kwargs
        )
