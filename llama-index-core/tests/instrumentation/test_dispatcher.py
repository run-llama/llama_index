import asyncio
import inspect
from asyncio import CancelledError
from collections import Counter

import pytest
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.dispatcher import Dispatcher
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from unittest.mock import patch, MagicMock

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
    events = []

    @classmethod
    def class_name(cls):
        return "_TestEventHandler"

    def handle(self, e: BaseEvent):
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
    dispatch_event = dispatcher.get_dispatch_event()

    dispatch_event(_TestStartEvent())


@dispatcher.span
async def async_func_with_event(a, b=3, **kwargs):
    dispatch_event = dispatcher.get_dispatch_event()

    dispatch_event(_TestStartEvent())
    await asyncio.sleep(0.1)
    dispatch_event(_TestEndEvent())


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
        dispatch_event = dispatcher.get_dispatch_event()

        dispatch_event(_TestStartEvent())

    @dispatcher.span
    async def async_func_with_event(self, a, b=3, **kwargs):
        dispatch_event = dispatcher.get_dispatch_event()

        dispatch_event(_TestStartEvent())
        await asyncio.sleep(0.1)
        await self.async_func(1)  # this should create a new span_id
        # that is fine because we have dispatch_event
        dispatch_event(_TestEndEvent())


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
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
@patch("llama_index.core.instrumentation.dispatcher.uuid")
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
@patch("llama_index.core.instrumentation.dispatcher.uuid")
def test_dispatcher_span_drop_args(
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    with pytest.raises(ValueError):
        # act
        _ = func_exc(3, b=5, c=2, d=5)

    # assert
    # span_enter
    mock_span_enter.assert_called_once()

    # span_drop
    mock_span_drop.assert_called_once()
    span_id = f"{func_exc.__qualname__}-mock"
    bound_args = inspect.signature(func_exc).bind(3, b=5, c=2, d=5)
    args, kwargs = mock_span_drop.call_args
    assert args == ()
    assert kwargs == {
        "id_": span_id,
        "bound_args": bound_args,
        "instance": None,
        "err": value_error,
    }

    # span_exit
    mock_span_exit.assert_not_called()


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
def test_dispatcher_span_drop_args(
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    with pytest.raises(ValueError):
        # act
        instance = _TestObject()
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


@pytest.mark.asyncio()
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
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


@pytest.mark.asyncio()
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
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


@pytest.mark.asyncio()
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
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


@pytest.mark.asyncio()
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
async def test_dispatcher_async_span_drop_args_with_instance(
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    with pytest.raises(CancelledError):
        # act
        instance = _TestObject()
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
@patch("llama_index.core.instrumentation.dispatcher.uuid")
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


@pytest.mark.asyncio()
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
    mock_span_enter.call_count == 3

    # span
    mock_span_drop.assert_not_called()

    # span_exit
    mock_span_exit.call_count == 3


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
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


@pytest.mark.asyncio()
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
    mock_span_enter.call_count == 2

    # span
    mock_span_drop.assert_not_called()

    # span_exit
    mock_span_exit.call_count == 2
