from typing import List, Optional, Dict
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
        default=NullSpanHandler(), description="Span handler."
    )
    parent_name: str = Field(
        default_factory=str, description="Name of parent Dispatcher."
    )
    manager: Optional["Manager"] = Field(
        default=None, description="Dispatcher manager."
    )
    root_name: str = Field(default="root", description="Root of the Dispatcher tree.")
    propagate: bool = Field(
        default=True,
        description="Whether to propagate the event to parent dispatchers and their handlers",
    )

    @property
    def parent(self) -> "Dispatcher":
        return self.manager.dispatchers[self.parent_name]

    @property
    def root(self) -> "Dispatcher":
        return self.manager.dispatchers[self.root_name]

    def add_event_handler(self, handler) -> None:
        """Add handler to set of handlers."""
        self.event_handlers += [handler]

    def event(self, event: BaseEvent, **kwargs) -> None:
        """Dispatch event to all registered handlers."""
        c = self
        while c:
            for h in c.event_handlers:
                h.handle(event, **kwargs)
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_enter(self, id: str) -> None:
        """Send notice to handlers that a span with id has started."""
        c = self
        while c:
            c.span_handler.span_enter(id)
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_drop(self, id: str, exception: Optional[Exception], **kwargs) -> None:
        """Send notice to handlers that a span with id is being dropped."""
        c = self
        while c:
            c.span_handler.span_drop(id, exception)
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_exit(self, id: str) -> None:
        """Send notice to handlers that a span with id is exiting."""
        c = self
        while c:
            c.span_handler.span_exit(id)
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            id = f"{func.__qualname__}-{uuid.uuid4()}"
            self.span_enter(id=id)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.span_drop(id=id, err=e)
            finally:
                self.span_exit(id=id)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            id = f"{func.__qualname__}-{uuid.uuid4()}"
            self.span_enter(id=id)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.span_drop(id=id, err=e)
            finally:
                self.span_exit(id=id)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    @property
    def log_name(self) -> str:
        """Name to be used in logging."""
        if self.parent:
            return f"{self.parent.name}.{self.name}"
        else:
            return self.name

    class Config:
        arbitrary_types_allowed = True


class Manager:
    def __init__(self, root: Dispatcher) -> None:
        self.dispatchers: Dict[str, Dispatcher] = {root.name: root}

    def add_dispatcher(self, d: Dispatcher) -> None:
        if d.name in self.dispatchers:
            pass
        else:
            self.dispatchers[d.name] = d


Dispatcher.update_forward_refs()
