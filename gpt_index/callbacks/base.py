import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional

from gpt_index.callbacks.schema import CBEvent, CBEventType


class BaseCallbackHandler(ABC):
    """Base callback handler that can be used to track event starts and ends."""
    def __init__(self, event_starts_to_ignore: List[CBEventType], event_ends_to_ignore: List[CBEventType]) -> None:
        """Initialize the base callback handler."""
        self.event_starts_to_ignore = tuple(event_starts_to_ignore)
        self.event_ends_to_ignore = tuple(event_ends_to_ignore)

    @abstractmethod
    def on_event_start(self, event: CBEvent, event_id: Optional[str] = None, **kwargs: Any) -> str:
        """Run when an event starts and return id of event."""
    
    @abstractmethod
    def on_event_end(self, event: CBEvent, event_id: Optional[str] = None, **kwargs: Any) -> None:
        """Run when an event ends."""


class CallbackManager(BaseCallbackHandler, ABC):
    """Callback manager that handles callbacks for events within LlamaIndex"""
    
    def __init__(self, handlers: List[BaseCallbackHandler]):
        """Initialize the manager with a list of handlers."""
        self.handlers = handlers
    
    def on_event_start(self, event: CBEvent, event_id: Optional[str] = None, **kwargs: Any) -> str:
        """Run handlers when an event starts and return id of event."""
        _id = event_id or str(uuid.uuid4())
        event.id = _id
        for handler in self.handlers:
            if event.event_type not in handler.event_starts_to_ignore:
                handler.on_event_start(event, **kwargs)
        return _id

    def on_event_end(self, event: CBEvent, event_id: Optional[str] = None, **kwargs: Any) -> None:
        """Run handlers when an event ends."""
        _id = event_id or str(uuid.uuid4())
        event.id = _id
        for handler in self.handlers:
            if event.event_type not in handler.event_ends_to_ignore:
                handler.on_event_start(event, **kwargs)

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = handlers
