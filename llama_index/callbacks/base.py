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
    def launch(self, run_id: Optional[str] = None) -> None:
        """Run when an overall run is launched."""

    @abstractmethod
    def shutdown(
        self,
        run_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall run is exited."""


class CallbackManager(BaseCallbackHandler, ABC):
    """Callback manager that handles callbacks for events within LlamaIndex.

    Args:
        handlers (List[BaseCallbackHandler]): list of handlers to use.

    """

    def __init__(self, handlers: List[BaseCallbackHandler]):
        """Initialize the manager with a list of handlers."""
        self.handlers = handlers
        self._trace_map: Dict[str, List[str]] = defaultdict(list)
        self._trace_stack: List[str] = [BASE_TRACE_ID]
        self._running_id: Optional[str] = None

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
    def as_trace(self, run_id: str) -> Generator[None, None, None]:
        """Context manager tracer for lanching and shutdown of runs."""
        self.launch(run_id=run_id)
        yield
        self.shutdown(run_id=run_id)

    def launch(self, run_id: Optional[str] = None) -> None:
        """Run when an overall run is launched."""
        if not self._running_id:
            self._trace_map = defaultdict(list)
            self._trace_stack = [BASE_TRACE_ID]

            for handler in self.handlers:
                handler.launch(run_id=run_id)

            self._running_id = run_id

    def shutdown(
        self,
        run_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall run is exited."""
        if run_id is not None and run_id == self._running_id:
            for handler in self.handlers:
                handler.shutdown(run_id=run_id, trace_map=self._trace_map)
            self._running_id = None

    @property
    def trace_map(self) -> Dict[str, List[str]]:
        return self._trace_map
