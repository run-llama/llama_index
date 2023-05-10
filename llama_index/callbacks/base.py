import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.callbacks.schema import CBEventType


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


class CallbackManager(BaseCallbackHandler, ABC):
    """Callback manager that handles callbacks for events within LlamaIndex.

    Args:
        handlers (List[BaseCallbackHandler]): list of handlers to use.

    """

    def __init__(self, handlers: List[BaseCallbackHandler]):
        """Initialize the manager with a list of handlers."""
        self.handlers = handlers

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> str:
        """Run handlers when an event starts and return id of event."""
        event_id = event_id or str(uuid.uuid4())
        for handler in self.handlers:
            if event_type not in handler.event_starts_to_ignore:
                handler.on_event_start(event_type, payload, event_id=event_id, **kwargs)
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

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = handlers
