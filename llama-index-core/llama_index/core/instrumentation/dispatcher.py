from typing import Any, List, Optional, Dict
import inspect
import uuid
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import (
    BaseSpanHandler,
    NullSpanHandler,
)
from llama_index.core.instrumentation.events.base import BaseEvent
import wrapt


class Dispatcher(BaseModel):
    name: str = Field(default_factory=str, description="Name of dispatcher")
    event_handlers: List[BaseEventHandler] = Field(
        default=[], description="List of attached handlers"
    )
    span_handlers: List[BaseSpanHandler] = Field(
        default=[NullSpanHandler()], description="Span handler."
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

    def add_event_handler(self, handler: BaseEventHandler) -> None:
        """Add handler to set of handlers."""
        self.event_handlers += [handler]

    def add_span_handler(self, handler: BaseSpanHandler) -> None:
        """Add handler to set of handlers."""
        self.span_handlers += [handler]

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

    def span_enter(
        self,
        *args: Any,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Send notice to handlers that a span with id_ has started."""
        c = self
        while c:
            for h in c.span_handlers:
                h.span_enter(
                    *args,
                    id_=id_,
                    bound_args=bound_args,
                    instance=instance,
                    **kwargs,
                )
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_drop(
        self,
        *args: Any,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> None:
        """Send notice to handlers that a span with id_ is being dropped."""
        c = self
        while c:
            for h in c.span_handlers:
                h.span_drop(
                    *args,
                    id_=id_,
                    bound_args=bound_args,
                    instance=instance,
                    err=err,
                    **kwargs,
                )
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_exit(
        self,
        *args: Any,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Send notice to handlers that a span with id_ is exiting."""
        c = self
        while c:
            for h in c.span_handlers:
                h.span_exit(
                    *args,
                    id_=id_,
                    bound_args=bound_args,
                    instance=instance,
                    result=result,
                    **kwargs,
                )
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span(self, func):
        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            id_ = f"{func.__qualname__}-{uuid.uuid4()}"
            self.span_enter(id_=id_, bound_args=bound_args, instance=instance)
            try:
                result = func(*args, **kwargs)
            except BaseException as e:
                self.span_drop(id_=id_, bound_args=bound_args, instance=instance, err=e)
                raise
            else:
                self.span_exit(
                    id_=id_, bound_args=bound_args, instance=instance, result=result
                )
                return result

        @wrapt.decorator
        async def async_wrapper(func, instance, args, kwargs):
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            id_ = f"{func.__qualname__}-{uuid.uuid4()}"
            self.span_enter(id_=id_, bound_args=bound_args, instance=instance)
            try:
                result = await func(*args, **kwargs)
            except BaseException as e:
                self.span_drop(id_=id_, bound_args=bound_args, instance=instance, err=e)
                raise
            else:
                self.span_exit(
                    id_=id_, bound_args=bound_args, instance=instance, result=result
                )
                return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper(func)
        else:
            return wrapper(func)

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
