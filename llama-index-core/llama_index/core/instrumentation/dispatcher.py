from typing import List, Optional, Type, Protocol
import functools
import inspect
import uuid
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import (
    BaseSpanHandler,
    NullSpanHandler,
)
from llama_index.core.instrumentation.events.base import BaseEvent


class Dispatcher(BaseModel):
    name: str = Field(default_factory=str, description="Name of dispatcher")
    event_handlers: List[BaseEventHandler] = Field(
        default=[], description="List of attached handlers"
    )
    span_handler: BaseSpanHandler = Field(
        default=NullSpanHandler, description="Span handler."
    )
    parent: Optional["Dispatcher"] = Field(
        default_factory=None, description="Optional parent Dispatcher"
    )
    propagate: bool = Field(
        default=True,
        description="Whether to propagate the event to parent dispatchers and their handlers",
    )

    def add_event_handler(self, handler) -> None:
        """Add handler to set of handlers."""
        self.event_handlers += [handler]

    def event(self, event_cls: Type[BaseEvent]) -> None:
        """Dispatch event to all registered handlers."""
        event = event_cls()
        for h in self.event_handlers:
            h.handle(event)

    def span_enter(self, id: str) -> None:
        """Send notice to handlers that a span with id has started."""
        self.span_handler.span_enter(id)

    def span_drop(self, id: str) -> None:
        """Send notice to handlers that a span with id has started."""
        return

    def span_exit(self, id: str) -> None:
        """Send notice to handlers that a span with id has started."""
        self.span_handler.span_exit(id)

    @property
    def log_name(self) -> str:
        """Name to be used in logging."""
        if self.parent:
            return f"{self.parent.name}.{self.name}"
        else:
            return self.name

    class Config:
        arbitrary_types_allowed = True


# class protocol
class HasDispatcherProtocol(Protocol):
    @property
    def dispatcher(self) -> Dispatcher:
        ...


class DispatcherMixin:
    @staticmethod
    def span(func):
        @functools.wraps(func)
        def wrapper(self: HasDispatcherProtocol, *args, **kwargs):
            id = f"{func.__name__}-{uuid.uuid4()}"
            self.dispatcher.span_enter(id=id)
            try:
                func(self, *args, **kwargs)
            except Exception as e:
                self.dispatcher.span_drop(id=id)
            finally:
                self.dispatcher.span_exit(id=id)

        @functools.wraps(func)
        async def async_wrapper(self: HasDispatcherProtocol, *args, **kwargs):
            id = f"{func.__name__}-{uuid.uuid4()}"
            self.dispatcher.span_enter(id=id)
            try:
                await func(self, *args, **kwargs)
            except Exception as e:
                self.dispatcher.span_drop(id=id)
            finally:
                self.dispatcher.span_exit(id=id)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
