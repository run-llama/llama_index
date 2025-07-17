import asyncio
import inspect
import threading
import time
from abc import abstractmethod
from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Queue,
    gather,
    get_event_loop,
    sleep,
)
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from random import random
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import llama_index_instrumentation as instrument
import pytest
import wrapt
from llama_index_instrumentation import DispatcherSpanMixin
from llama_index_instrumentation.base import BaseEvent
from llama_index_instrumentation.dispatcher import Dispatcher, instrument_tags
from llama_index_instrumentation.event_handlers import BaseEventHandler
from llama_index_instrumentation.span import BaseSpan
from llama_index_instrumentation.span_handlers import BaseSpanHandler
from llama_index_instrumentation.span_handlers.base import Thread

dispatcher = instrument.get_dispatcher("test")

value_error = ValueError("value error")
cancelled_error = CancelledError("cancelled error")


class _TestStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        return "_TestStartEvent"


class _TestEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        return "_TestEndEvent"


class _TestEventHandler(BaseEventHandler):
    events: List[BaseEvent] = []

    @classmethod
    def class_name(cls):
        return "_TestEventHandler"

    def handle(self, e: BaseEvent):  # type:ignore
        self.events.append(e)


@dispatcher.span
def func(a, b=3, **kwargs):
    return a + b


@dispatcher.span
async def async_func(a, b=3, **kwargs):
    return a + b


@dispatcher.span
def func_exc(a, b=3, c=4, **kwargs):
    raise value_error


@dispatcher.span
async def async_func_exc(a, b=3, c=4, **kwargs):
    raise cancelled_error


@dispatcher.span
def func_with_event(a, b=3, **kwargs):
    dispatcher.event(_TestStartEvent())


@dispatcher.span
async def async_func_with_event(a, b=3, **kwargs):
    dispatcher.event(_TestStartEvent())
    await asyncio.sleep(0.1)
    dispatcher.event(_TestEndEvent())


# Can remove this test once dispatcher.get_dispatch_event is safely dopped.
@dispatcher.span
def func_with_event_backwards_compat(a, b=3, **kwargs):
    dispatch_event = dispatcher.get_dispatch_event()
    dispatch_event(_TestStartEvent())


class _TestObject:
    @dispatcher.span
    def func(self, a, b=3, **kwargs):
        return a + b

    @dispatcher.span
    async def async_func(self, a, b=3, **kwargs):
        return a + b

    @dispatcher.span
    def func_exc(self, a, b=3, c=4, **kwargs):
        raise value_error

    @dispatcher.span
    async def async_func_exc(self, a, b=3, c=4, **kwargs):
        raise cancelled_error

    @dispatcher.span
    def func_with_event(self, a, b=3, **kwargs):
        dispatcher.event(_TestStartEvent())

    @dispatcher.span
    async def async_func_with_event(self, a, b=3, **kwargs):
        dispatcher.event(_TestStartEvent())
        await asyncio.sleep(0.1)
        await self.async_func(1)  # this should create a new span_id
        # that is fine because we have dispatch_event
        dispatcher.event(_TestEndEvent())

    # Can remove this test once dispatcher.get_dispatch_event is safely dopped.
    @dispatcher.span
    def func_with_event_backwards_compat(self, a, b=3, **kwargs):
        dispatch_event = dispatcher.get_dispatch_event()
        dispatch_event(_TestStartEvent())


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_dispatcher_span_args(mock_uuid, mock_span_enter, mock_span_exit):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    # act
    result = func(3, c=5)

    # assert
    # span_enter
    span_id = f"{func.__qualname__}-mock"
    bound_args = inspect.signature(func).bind(3, c=5)
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": None,
        "parent_id": None,
        "tags": {},
    }

    # span_exit
    args, kwargs = mock_span_exit.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": None,
        "result": result,
    }


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_dispatcher_span_args_with_instance(mock_uuid, mock_span_enter, mock_span_exit):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    # act
    instance = _TestObject()
    result = instance.func(3, c=5)

    # assert
    # span_enter
    span_id = f"{instance.func.__qualname__}-mock"
    bound_args = inspect.signature(instance.func).bind(3, c=5)
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": instance,
        "parent_id": None,
        "tags": {},
    }

    # span_exit
    args, kwargs = mock_span_exit.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": instance,
        "result": result,
    }


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_dispatcher_span_drop_args(
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    instance = _TestObject()
    with pytest.raises(ValueError):
        _ = instance.func_exc(a=3, b=5, c=2, d=5)

    # assert
    # span_enter
    mock_span_enter.assert_called_once()

    # span_drop
    mock_span_drop.assert_called_once()
    span_id = f"{instance.func_exc.__qualname__}-mock"
    bound_args = inspect.signature(instance.func_exc).bind(a=3, b=5, c=2, d=5)
    args, kwargs = mock_span_drop.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": instance,
        "err": value_error,
    }

    # span_exit
    mock_span_exit.assert_not_called()


@pytest.mark.asyncio
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
async def test_dispatcher_async_span_args(mock_uuid, mock_span_enter, mock_span_exit):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    # act
    result = await async_func(a=3, c=5)

    # assert
    # span_enter
    span_id = f"{async_func.__qualname__}-mock"
    bound_args = inspect.signature(async_func).bind(a=3, c=5)
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": None,
        "parent_id": None,
        "tags": {},
    }

    # span_exit
    args, kwargs = mock_span_exit.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": None,
        "result": result,
    }


@pytest.mark.asyncio
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
async def test_dispatcher_async_span_args_with_instance(
    mock_uuid, mock_span_enter, mock_span_exit
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    # act
    instance = _TestObject()
    result = await instance.async_func(a=3, c=5)

    # assert
    # span_enter
    span_id = f"{instance.async_func.__qualname__}-mock"
    bound_args = inspect.signature(instance.async_func).bind(a=3, c=5)
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": instance,
        "parent_id": None,
        "tags": {},
    }

    # span_exit
    args, kwargs = mock_span_exit.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": instance,
        "result": result,
    }


@pytest.mark.asyncio
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
async def test_dispatcher_async_span_drop_args(
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    with pytest.raises(CancelledError):
        # act
        _ = await async_func_exc(a=3, b=5, c=2, d=5)

    # assert
    # span_enter
    mock_span_enter.assert_called_once()

    # span_drop
    mock_span_drop.assert_called_once()
    span_id = f"{async_func_exc.__qualname__}-mock"
    bound_args = inspect.signature(async_func_exc).bind(a=3, b=5, c=2, d=5)
    args, kwargs = mock_span_drop.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": None,
        "err": cancelled_error,
    }

    # span_exit
    mock_span_exit.assert_not_called()


@pytest.mark.asyncio
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
async def test_dispatcher_async_span_drop_args_with_instance(
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    instance = _TestObject()
    with pytest.raises(CancelledError):
        _ = await instance.async_func_exc(a=3, b=5, c=2, d=5)

    # assert
    # span_enter
    mock_span_enter.assert_called_once()

    # span_drop
    mock_span_drop.assert_called_once()
    span_id = f"{instance.async_func_exc.__qualname__}-mock"
    bound_args = inspect.signature(instance.async_func_exc).bind(a=3, b=5, c=2, d=5)
    args, kwargs = mock_span_drop.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": instance,
        "err": cancelled_error,
    }

    # span_exit
    mock_span_exit.assert_not_called()


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_dispatcher_fire_event(
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)

    # act
    _ = func_with_event(3, c=5)

    # assert
    span_id = f"{func_with_event.__qualname__}-mock"
    assert all(e.span_id == span_id for e in event_handler.events)

    # span_enter
    mock_span_enter.assert_called_once()

    # span
    mock_span_drop.assert_not_called()

    # span_exit
    mock_span_exit.assert_called_once()


@pytest.mark.asyncio
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
async def test_dispatcher_async_fire_event(
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)

    # act
    tasks = [
        async_func_with_event(a=3, c=5),
        async_func_with_event(5),
        async_func_with_event(4),
    ]
    _ = await asyncio.gather(*tasks)

    # assert
    span_ids = [e.span_id for e in event_handler.events]
    id_counts = Counter(span_ids)
    assert set(id_counts.values()) == {2}

    # span_enter
    assert mock_span_enter.call_count == 3

    # span
    mock_span_drop.assert_not_called()

    # span_exit
    assert mock_span_exit.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
@patch.object(Dispatcher, "span_enter")
async def test_dispatcher_attaches_tags_to_events_and_spans(
    mock_span_enter: MagicMock,
    use_async: bool,
):
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)
    test_tags = {"test_tag_key": "test_tag_value"}

    # Check that tags are set when using context manager
    with instrument_tags(test_tags):
        if use_async:
            await async_func_with_event(a=3, c=5)
        else:
            func_with_event(a=3, c=5)

    mock_span_enter.assert_called_once()
    assert mock_span_enter.call_args[1]["tags"] == test_tags
    assert all(e.tags == test_tags for e in event_handler.events)


@patch.object(Dispatcher, "span_enter")
def test_dispatcher_attaches_tags_to_concurrent_events(
    mock_span_enter: MagicMock,
):
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)

    num_functions = 5
    test_tags = [{"test_tag_key": num} for num in range(num_functions)]
    test_tags_set = {str(tag) for tag in test_tags}

    def run_func_with_tags(tag):
        with instrument_tags(tag):
            func_with_event(3, c=5)

    # Run functions concurrently
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for tag in test_tags:
            futures.append(executor.submit(run_func_with_tags, tag))

    for future in futures:
        future.result()

    # Ensure that each function recorded a span and event with the tags
    assert len(mock_span_enter.call_args_list) == num_functions
    assert len(event_handler.events) == num_functions
    actual_span_tags = {
        str(call_kwargs["tags"]) for _, call_kwargs in mock_span_enter.call_args_list
    }
    actual_event_tags = {str(e.tags) for e in event_handler.events}
    assert actual_span_tags == test_tags_set
    assert actual_event_tags == test_tags_set


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_dispatcher_fire_event_with_instance(
    mock_uuid, mock_span_enter, mock_span_drop, mock_span_exit
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)

    # act
    instance = _TestObject()
    _ = instance.func_with_event(a=3, c=5)

    # assert
    span_id = f"{instance.func_with_event.__qualname__}-mock"
    assert all(e.span_id == span_id for e in event_handler.events)

    # span_enter
    mock_span_enter.assert_called_once()

    # span
    mock_span_drop.assert_not_called()

    # span_exit
    mock_span_exit.assert_called_once()


@pytest.mark.asyncio
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
async def test_dispatcher_async_fire_event_with_instance(
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    # mock_uuid.return_value = "mock"
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)

    # act
    instance = _TestObject()
    tasks = [
        instance.async_func_with_event(a=3, c=5),
        instance.async_func_with_event(5),
    ]
    _ = await asyncio.gather(*tasks)

    # assert
    span_ids = [e.span_id for e in event_handler.events]
    id_counts = Counter(span_ids)
    assert set(id_counts.values()) == {2}

    # span_enter
    assert mock_span_enter.call_count == 4

    # span
    mock_span_drop.assert_not_called()

    # span_exit
    assert mock_span_exit.call_count == 4


def test_context_nesting():
    # arrange
    # A binary tree of parent-child spans
    h = 5  # height of binary tree
    s = 2 ** (h + 1) - 1  # number of spans per tree
    runs = 2  # number of trees (in parallel)
    # Below is a tree (r=1) with h=3 (s=15).
    # Tn: n-th span run in thread
    # An: n-th span run in async
    #               A1
    #       ┌───────┴───────┐
    #       A2              A3
    #   ┌───┴───┐       ┌───┴───┐
    #   T4      T5      A6      A7
    # ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
    # T8  T9  A10 A11 T12 T13 A14 A15
    # Note that child.n // 2 == parent.n, e.g. 11 // 2 == 5.
    # We'll check that the parent-child associations are correct.

    class Span(BaseSpan):
        r: int  # tree id
        n: int  # span id (per tree)

    class Event(BaseEvent):
        r: int  # tree id
        n: int  # span id (per tree)

    lock = Lock()
    spans: Dict[str, Span] = {}
    events: List[Event] = []

    class SpanHandler(BaseSpanHandler):
        def new_span(
            self,
            id_: str,
            bound_args: inspect.BoundArguments,
            instance: Optional[Any] = None,
            parent_span_id: Optional[str] = None,
            tags: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            r, n = bound_args.args[:2]
            span = Span(r=r, n=n, id_=id_, parent_id=parent_span_id)
            with lock:
                spans[id_] = span

        def prepare_to_drop_span(self, *args: Any, **kwargs: Any) -> None: ...

        def prepare_to_exit_span(self, *args: Any, **kwargs: Any) -> None: ...

    class EventHandler(BaseEventHandler):
        def handle(self, event: Event, **kwargs) -> None:  # type: ignore
            with lock:
                events.append(event)

    dispatcher = Dispatcher(
        event_handlers=[EventHandler()],
        span_handlers=[SpanHandler()],
        propagate=False,
    )

    @dispatcher.span
    def bar(r: int, n: int, callback: Callable[[], None] = lambda: None) -> None:
        dispatcher.event(Event(r=r, n=n))
        if n > 2**h - 1:
            callback()
            return
        if n % 2:
            asyncio.run(_foo(r, n))
        else:
            t0 = Thread(target=bar, args=(r, n * 2))
            t1 = Thread(target=bar, args=(r, n * 2 + 1))
            t0.start()
            t1.start()
            time.sleep(0.01)
            t0.join()
            t1.join()
        callback()

    @dispatcher.span
    async def foo(r: int, n: int) -> None:
        dispatcher.event(Event(r=r, n=n))
        if n > 2**h - 1:
            return
        if n % 2:
            await _foo(r, n)
        else:
            q, loop = Queue(), get_event_loop()
            Thread(target=bar, args=(r, n * 2, _callback(q, loop))).start()
            Thread(target=bar, args=(r, n * 2 + 1, _callback(q, loop))).start()
            await gather(q.get(), q.get())

    async def _foo(r: int, n: int) -> None:
        await gather(foo(r, n * 2), foo(r, n * 2 + 1), sleep(0.01))

    def _callback(q: Queue, loop: AbstractEventLoop) -> Callable[[], None]:
        return lambda: loop.call_soon_threadsafe(q.put_nowait(1))  # type: ignore

    # act
    # Use regular thread to ensure that `Token.MISSING` is being handled.
    regular_threads = [
        (
            threading.Thread(target=asyncio.run, args=(foo(r, 1),))
            if r % 2
            else threading.Thread(target=bar, args=(r, 1))
        )
        for r in range(runs)
    ]
    [t.start() for t in regular_threads]
    [t.join() for t in regular_threads]

    # assert
    # parent-child associations should be correct
    assert sorted(span.n for span in spans.values()) == sorted(
        list(range(1, s + 1)) * runs
    )
    for span in spans.values():
        if span.n > 1:
            if not span.parent_id:
                print(span)
            assert span.r == spans[span.parent_id].r  # same tree  #type:ignore
            assert span.n // 2 == spans[span.parent_id].n  # type:ignore

    # # event-span associations should be correct
    # assert sorted(event.n for event in events) == sorted(list(range(1, s + 1)) * runs)
    # for event in events:
    #     assert event.r == spans[event.span_id].r  # same tree
    #     assert event.n == spans[event.span_id].n  # same span


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_dispatcher_fire_event_backwards_compat(
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)

    # act
    _ = func_with_event_backwards_compat(3, c=5)

    # assert
    span_id = f"{func_with_event_backwards_compat.__qualname__}-mock"
    assert all(e.span_id == span_id for e in event_handler.events)

    # span_enter
    mock_span_enter.assert_called_once()

    # span
    mock_span_drop.assert_not_called()

    # span_exit
    mock_span_exit.assert_called_once()


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_dispatcher_fire_event_with_instance_backwards_compat(
    mock_uuid, mock_span_enter, mock_span_drop, mock_span_exit
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"
    event_handler = _TestEventHandler()
    dispatcher.add_event_handler(event_handler)

    # act
    instance = _TestObject()
    _ = instance.func_with_event_backwards_compat(a=3, c=5)

    # assert
    span_id = f"{instance.func_with_event_backwards_compat.__qualname__}-mock"
    assert all(e.span_id == span_id for e in event_handler.events)

    # span_enter
    mock_span_enter.assert_called_once()

    # span
    mock_span_drop.assert_not_called()

    # span_exit
    mock_span_exit.assert_called_once()


@patch.object(Dispatcher, "span_enter")
def test_span_decorator_is_idempotent(mock_span_enter):
    x, z = random(), dispatcher.span
    assert z(z(z(lambda: x)))() == x
    mock_span_enter.assert_called_once()


@patch.object(Dispatcher, "span_enter")
def test_span_decorator_is_idempotent_with_pass_through(mock_span_enter):
    x, z = random(), dispatcher.span
    a, b, c, d = (wrapt.decorator(lambda f, *_: f()) for _ in range(4))
    assert z(a(b(z(c(d(z(lambda: x)))))))() == x
    mock_span_enter.assert_called_once()


@patch.object(Dispatcher, "span_enter")
def test_mixin_decorates_abstract_method(mock_span_enter):
    x, z = random(), abstractmethod
    A = type("A", (DispatcherSpanMixin,), {"f": z(lambda _: ...)})
    B = type("B", (A,), {"f": lambda _: x + 0})
    C = type("C", (B,), {"f": lambda _: x + 1})
    D = type("D", (C, B), {"f": lambda _: x + 2})
    for i, T in enumerate((B, C, D)):
        assert T().f() - i == pytest.approx(x)  # type:ignore
        assert mock_span_enter.call_count - i == 1


@patch.object(Dispatcher, "span_enter")
def test_mixin_decorates_overridden_method(mock_span_enter):
    x, z = random(), dispatcher.span
    A = type("A", (DispatcherSpanMixin,), {"f": z(lambda _: x)})
    B = type("B", (A,), {"f": lambda _: x + 1})
    C = type("C", (B,), {"f": lambda _: x + 2})
    D = type("D", (C, B), {"f": lambda _: x + 3})
    for i, T in enumerate((A, B, C, D)):
        assert T().f() - i == pytest.approx(x)  # type:ignore
        assert mock_span_enter.call_count - i == 1


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_span_naming_with_inheritance(mock_uuid, mock_span_enter, mock_span_exit):
    """Test that span IDs use the runtime class name, not the definition class name."""
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    class BaseClass:
        @dispatcher.span
        def base_method(self, x):
            return x * 2

        @dispatcher.span
        async def async_base_method(self, x):
            return x * 3

    class DerivedClass(BaseClass):
        pass

    class AnotherDerivedClass(BaseClass):
        @dispatcher.span
        def derived_method(self, x):
            return x * 4

    # act
    base_instance = BaseClass()
    derived_instance = DerivedClass()
    another_derived_instance = AnotherDerivedClass()

    base_result = base_instance.base_method(5)
    derived_result = derived_instance.base_method(5)
    another_derived_result = another_derived_instance.derived_method(5)

    # assert
    assert mock_span_enter.call_count == 3

    # Check that span IDs use the actual runtime class names
    calls = mock_span_enter.call_args_list

    # BaseClass.base_method called on BaseClass instance
    assert calls[0][1]["id_"] == "BaseClass.base_method-mock"

    # BaseClass.base_method called on DerivedClass instance (should use DerivedClass)
    assert calls[1][1]["id_"] == "DerivedClass.base_method-mock"

    # AnotherDerivedClass.derived_method called on AnotherDerivedClass instance
    assert calls[2][1]["id_"] == "AnotherDerivedClass.derived_method-mock"


@pytest.mark.asyncio
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
async def test_async_span_naming_with_inheritance(
    mock_uuid, mock_span_enter, mock_span_exit
):
    """Test that async span IDs use the runtime class name, not the definition class name."""
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    class BaseClass:
        @dispatcher.span
        async def async_base_method(self, x):
            return x * 3

    class DerivedClass(BaseClass):
        pass

    # act
    base_instance = BaseClass()
    derived_instance = DerivedClass()

    base_result = await base_instance.async_base_method(5)
    derived_result = await derived_instance.async_base_method(5)

    # assert
    assert mock_span_enter.call_count == 2

    calls = mock_span_enter.call_args_list

    # BaseClass.async_base_method called on BaseClass instance
    assert calls[0][1]["id_"] == "BaseClass.async_base_method-mock"

    # BaseClass.async_base_method called on DerivedClass instance (should use DerivedClass)
    assert calls[1][1]["id_"] == "DerivedClass.async_base_method-mock"


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_span_naming_regular_functions_unchanged(
    mock_uuid, mock_span_enter, mock_span_exit
):
    """Test that regular functions (non-methods) still use __qualname__."""
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    @dispatcher.span
    def regular_function(x):
        return x * 5

    # act
    result = regular_function(10)

    # assert
    mock_span_enter.assert_called_once()
    call_kwargs = mock_span_enter.call_args[1]

    # Regular functions should still use __qualname__
    assert call_kwargs["id_"] == f"{regular_function.__qualname__}-mock"


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_span_naming_complex_inheritance(mock_uuid, mock_span_enter, mock_span_exit):
    """Test span naming with multiple levels of inheritance."""
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    class GrandParent:
        @dispatcher.span
        def shared_method(self, x):
            return x

    class Parent(GrandParent):
        pass

    class Child(Parent):
        @dispatcher.span
        def child_method(self, x):
            return x * 2

    class GrandChild(Child):
        pass

    # act
    instances = [GrandParent(), Parent(), Child(), GrandChild()]

    # Call shared_method on all instances
    for instance in instances:
        instance.shared_method(1)

    # Call child_method on child and grandchild
    instances[2].child_method(1)  # Child
    instances[3].child_method(1)  # GrandChild

    # assert
    assert mock_span_enter.call_count == 6

    calls = mock_span_enter.call_args_list

    # shared_method calls should use the runtime class names
    assert calls[0][1]["id_"] == "GrandParent.shared_method-mock"
    assert calls[1][1]["id_"] == "Parent.shared_method-mock"
    assert calls[2][1]["id_"] == "Child.shared_method-mock"
    assert calls[3][1]["id_"] == "GrandChild.shared_method-mock"

    # child_method calls should use the runtime class names
    assert calls[4][1]["id_"] == "Child.child_method-mock"
    assert calls[5][1]["id_"] == "GrandChild.child_method-mock"


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_span_naming_with_method_override(mock_uuid, mock_span_enter, mock_span_exit):
    """Test span naming when methods are overridden in derived classes."""
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    class Base:
        @dispatcher.span
        def method(self, x):
            return x

    class Derived(Base):
        @dispatcher.span
        def method(self, x):
            return x * 2

    # act
    base_instance = Base()
    derived_instance = Derived()

    base_instance.method(1)
    derived_instance.method(1)

    # assert
    assert mock_span_enter.call_count == 2

    calls = mock_span_enter.call_args_list

    # Each should use their respective class names
    assert calls[0][1]["id_"] == "Base.method-mock"
    assert calls[1][1]["id_"] == "Derived.method-mock"


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index_instrumentation.dispatcher.uuid")
def test_span_naming_with_nested_classes(mock_uuid, mock_span_enter, mock_span_exit):
    """Test span naming with nested classes."""
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    class Outer:
        class Inner:
            @dispatcher.span
            def inner_method(self, x):
                return x

        @dispatcher.span
        def outer_method(self, x):
            return x * 2

    # act
    outer_instance = Outer()
    inner_instance = Outer.Inner()

    outer_instance.outer_method(1)
    inner_instance.inner_method(1)

    # assert
    assert mock_span_enter.call_count == 2

    calls = mock_span_enter.call_args_list

    # Should use the simple class names (not qualified names)
    assert calls[0][1]["id_"] == "Outer.outer_method-mock"
    assert calls[1][1]["id_"] == "Inner.inner_method-mock"
