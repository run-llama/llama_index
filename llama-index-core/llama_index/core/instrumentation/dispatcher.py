from typing import Any, List, Optional, Dict, Protocol
from functools import partial
from contextlib import contextmanager
import asyncio
import inspect
import uuid
from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import (
    BaseSpanHandler,
    NullSpanHandler,
)
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.instrumentation.events.span import SpanDropEvent
from contextvars import ContextVar
import wrapt


span_ctx = ContextVar("span_ctx", default={})


def get_await_stack(coro):
    """Return a coroutine's chain of awaiters.

    This follows the cr_await links.
    """
    stack = []
    while coro is not None and hasattr(coro, "cr_await"):
        stack.append(coro)
        coro = coro.cr_await
    return stack


def get_task_tree():
    """Return the task tree dict {awaited: awaiting}.

    This follows the _fut_waiter links and constructs a map
    from awaited tasks to the tasks that await them.
    """
    tree = {}
    for task in asyncio.all_tasks():
        awaited = task._fut_waiter
        if awaited is not None:
            tree[awaited] = task
    return tree


def get_task_stack(task):
    """Return the stack of tasks awaiting a task.

    For each task it returns a tuple (task, awaiters) where
    awaiters is the chain of coroutines comprising the task.
    The first entry is the argument task, the last entry is
    the root task (often "Task-1", created by asyncio.run()).

    For example, if we have a task A running a coroutine f1,
    where f1 awaits f2, and f2 awaits a task B running a coroutine
    f3 which awaits f4 which awaits f5, then we'll return

        [
            (B, [f3, f4, f5]),
            (A, [f1, f2]),
        ]

    NOTE: The coroutine stack for the *current* task is inaccessible.
    To work around this, use `await task_stack()`.

    Todo:
    - Maybe it would be nicer to reverse the stack?
    - Classic coroutines and async generators are not supported yet.
    - This is expensive due to the need to first create a reverse
      mapping of awaited tasks to awaiting tasks.
    """
    tree = get_task_tree()
    stack = []
    while task is not None:
        coro = task.get_coro()
        awaiters = get_await_stack(coro)
        stack.append((task, awaiters))
        task = tree.get(task)
    return stack


async def async_task_stack():
    """Return the stack of tasks awaiting the current task.

    This exists so you can get the coroutine stack for the current task.
    """
    task = asyncio.current_task()

    async def helper(task):
        return get_task_stack(task)

    return await asyncio.create_task(helper(task), name="TaskStackHelper")


class EventDispatcher(Protocol):
    def __call__(self, event: BaseEvent) -> None:
        ...


class EventContext(BaseModel):
    span_id: str = Field(default="")


event_context = EventContext()


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
    current_span_id: Optional[str] = Field(
        default=None, description="Id of current span."
    )
    _asyncio_lock: asyncio.Lock = PrivateAttr()

    def __init__(
        self,
        name: str = "",
        event_handlers: List[BaseEventHandler] = [],
        span_handlers: List[BaseSpanHandler] = [],
        parent_name: str = "",
        manager: Optional["Manager"] = None,
        root_name: str = "root",
        propagate: bool = True,
    ):
        self._asyncio_lock = asyncio.Lock()
        super().__init__(
            name=name,
            event_handlers=event_handlers,
            span_handlers=span_handlers,
            parent_name=parent_name,
            manager=manager,
            root_name=root_name,
            propagate=propagate,
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

    def event(self, event: BaseEvent, span_id: Optional[str] = None, **kwargs) -> None:
        """Dispatch event to all registered handlers."""
        c = self
        if span_id:
            event.span_id = span_id
        while c:
            for h in c.event_handlers:
                h.handle(event, **kwargs)
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_enter(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Send notice to handlers that a span with id_ has started."""
        c = self
        while c:
            for h in c.span_handlers:
                h.span_enter(
                    id_=id_,
                    bound_args=bound_args,
                    instance=instance,
                    parent_id=parent_id,
                    **kwargs,
                )
            if not c.propagate:
                c = None
            else:
                c = c.parent

    def span_drop(
        self,
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

    def get_dispatch_event(self) -> EventDispatcher:
        """Get dispatch_event for firing events within the context of a span.

        This method should be used with @dispatcher.span decorated
        functions only. Otherwise, the span_id should not be trusted, as the
        span decorator sets the span_id.
        """
        span_id = self.current_span_id
        dispatch_event: EventDispatcher = partial(self.event, span_id=span_id)
        return dispatch_event

    @contextmanager
    def dispatch_event(self):
        """Context manager for firing events within a span session.

        This context manager should be used with @dispatcher.span decorated
        functions only. Otherwise, the span_id should not be trusted, as the
        span decorator sets the span_id.
        """
        span_id = self.current_span_id
        dispatch_event: EventDispatcher = partial(self.event, span_id=span_id)

        try:
            yield dispatch_event
        finally:
            del dispatch_event

    def span_with_parent_id(self, parent_id):
        def outer(func):
            @wrapt.decorator
            async def async_wrapper(func, instance, args, kwargs):
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                id_ = f"{func.__qualname__}-{uuid.uuid4()}"
                async with self._asyncio_lock:
                    self.current_span_id = id_
                async with self.root._asyncio_lock:
                    self.root.current_span_id = id_

                task_stack = await async_task_stack()
                current_task, _awaiters = task_stack[0]
                current_task_name = current_task.get_name()
                span_ctx_dict = span_ctx.get().copy()
                if current_task_name not in span_ctx_dict:
                    span_ctx_dict[current_task_name] = [id_]
                else:
                    span_ctx_dict[current_task_name].append(id_)
                span_ctx.set(span_ctx_dict)

                self.span_enter(
                    id_=id_,
                    bound_args=bound_args,
                    instance=instance,
                    parent_id=parent_id,
                )
                try:
                    # print("\n")
                    # print(f"CURRENT TASKS:\n\n")
                    # for t in task_stack:
                    #     print(f"{t}\n")
                    # print("\n")
                    coro = func(*args, **kwargs)
                    print(f"CURRENT TASK NAME: {current_task.get_name()}\n")
                    print(f"CURRENT CORO: {coro}\n\n")
                    result = await coro
                except BaseException as e:
                    self.event(SpanDropEvent(span_id=id_, err_str=str(e)))
                    self.span_drop(
                        id_=id_, bound_args=bound_args, instance=instance, err=e
                    )
                    raise
                else:
                    self.span_exit(
                        id_=id_, bound_args=bound_args, instance=instance, result=result
                    )
                    return result
                finally:
                    # clean up
                    task_stack = await async_task_stack()
                    current_task, _awaiters = task_stack[0]
                    current_task_name = current_task.get_name()
                    span_ctx_dict = span_ctx.get().copy()
                    span_ctx_dict[current_task_name].pop()
                    if len(span_ctx_dict[current_task_name]) == 0:
                        del span_ctx_dict[current_task_name]
                    span_ctx.set(span_ctx_dict)

            return async_wrapper(func)

        return outer

    def span(self, func):
        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            id_ = f"{func.__qualname__}-{uuid.uuid4()}"
            self.current_span_id = id_
            self.root.current_span_id = id_
            self.span_enter(id_=id_, bound_args=bound_args, instance=instance)
            try:
                result = func(*args, **kwargs)
            except BaseException as e:
                self.event(SpanDropEvent(span_id=id_, err_str=str(e)))
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
            async with self._asyncio_lock:
                self.current_span_id = id_
            async with self.root._asyncio_lock:
                self.root.current_span_id = id_

            # get parent_id
            task_stack = await async_task_stack()
            current_task, _awaiters = task_stack[0]
            current_task_name = current_task.get_name()
            span_ctx_dict = span_ctx.get().copy()
            if current_task_name not in span_ctx_dict:
                parent_id = None
                span_ctx_dict[current_task_name] = [id_]
            else:
                parent_id = span_ctx_dict[current_task_name][-1]
                span_ctx_dict[current_task_name].append(id_)
            span_ctx.set(span_ctx_dict)

            self.span_enter(
                id_=id_, bound_args=bound_args, instance=instance, parent_id=parent_id
            )
            try:
                # print("\n")
                # print(f"CURRENT TASKS:\n\n")
                # for t in task_stack:
                #     print(f"{t}\n")
                # print("\n")
                coro = func(*args, **kwargs)
                print(f"CURRENT TASK NAME: {current_task.get_name()}\n")
                print(f"CURRENT CORO: {coro}\n\n")
                result = await coro
            except BaseException as e:
                self.event(SpanDropEvent(span_id=id_, err_str=str(e)))
                self.span_drop(id_=id_, bound_args=bound_args, instance=instance, err=e)
                raise
            else:
                self.span_exit(
                    id_=id_, bound_args=bound_args, instance=instance, result=result
                )
                return result
            finally:
                # clean up
                task_stack = await async_task_stack()
                current_task, _awaiters = task_stack[0]
                current_task_name = current_task.get_name()
                span_ctx_dict = span_ctx.get().copy()
                span_ctx_dict[current_task_name].pop()
                if len(span_ctx_dict[current_task_name]) == 0:
                    del span_ctx_dict[current_task_name]
                span_ctx.set(span_ctx_dict)

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
