from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from gpt_index.callback_manager.schema import CBEvent, CBEventType


class BaseCallbackHandler(ABC):
    """Base callback handler that can be used to track event starts and ends."""
    
    @property
    def event_starts_to_ignore(self) -> Tuple[CBEventType]:
        return Tuple()
    
    @property
    def event_ends_to_ignore(self) -> Tuple[CBEventType]:
        return Tuple()

    @abstractmethod
    def on_event_start(self, event_type: CBEventType, event: CBEvent, **kwargs: Any):
        """Run when an event starts."""
    
    @abstractmethod
    def on_event_end(self, event_type: CBEventType, event: CBEvent, **kwargs: Any):
        """Run when an event ends."""


class BaseCallbackManager(BaseCallbackHandler, ABC):
    """Base callback manager that handles callbacks for events within LlamaIndex"""

    @abstractmethod
    def add_handler(self, callback: BaseCallbackHandler) -> None:
        """Add a callback handler to the manager."""
    
    @abstractmethod
    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the manager."""

    def set_handler(self, handler: BaseCallbackHandler) -> None:
        """Set handlers list to a specific single handler."""
        self.set_handlers([handler])

    @abstractmethod
    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers list to a specific set of handlers."""


class CallbackManager(BaseCallbackManager):
    """Callback manager that handles callbacks for events within LlamaIndex"""
    
    def __init__(self, handlers: List[BaseCallbackHandler]):
        """Initialize the manager with a list of handlers."""
        self.handlers = handlers
    
    def on_event_start(self, event_type: CBEventType, event: CBEvent, **kwargs: Any):
        """Run handlers when an event starts."""
        for handler in self.handlers:
            if event_type not in handler.event_starts_to_ignore:
                handler.on_event_start(event_type, event, **kwargs)

    def on_event_end(self, event_type: CBEventType, event: CBEvent, **kwargs: Any):
        """Run handlers when an event ends."""
        for handler in self.handlers:
            if event_type not in handler.event_ends_to_ignore:
                handler.on_event_start(event_type, event, **kwargs)

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = handlers
