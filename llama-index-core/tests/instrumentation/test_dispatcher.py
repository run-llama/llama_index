import inspect
from asyncio import CancelledError

import pytest
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.dispatcher import Dispatcher
from unittest.mock import patch, MagicMock

dispatcher = instrument.get_dispatcher("test")

value_error = ValueError("value error")
cancelled_error = CancelledError("cancelled error")


@dispatcher.span
def func(*args, a, b=3, **kwargs):
    return a + b


@dispatcher.span
async def async_func(*args, a, b=3, **kwargs):
    return a + b


@dispatcher.span
def func_exc(*args, a, b=3, c=4, **kwargs):
    raise value_error


@dispatcher.span
async def async_func_exc(*args, a, b=3, c=4, **kwargs):
    raise cancelled_error


class _TestObject:
    @dispatcher.span
    def func(self, *args, a, b=3, **kwargs):
        return a + b

    @dispatcher.span
    async def async_func(self, *args, a, b=3, **kwargs):
        return a + b

    @dispatcher.span
    def func_exc(self, *args, a, b=3, c=4, **kwargs):
        raise value_error

    @dispatcher.span
    async def async_func_exc(self, *args, a, b=3, c=4, **kwargs):
        raise cancelled_error


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
def test_dispatcher_span_args(mock_uuid, mock_span_enter, mock_span_exit):
    # arrange
    mock_uuid.uuid4.return_value = "mock"

    # act
    result = func(1, 2, a=3, c=5)

    # assert
    # span_enter
    span_id = f"{func.__qualname__}-mock"
    bound_args = inspect.signature(func).bind(1, 2, a=3, c=5)
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == ()
    assert kwargs == {"id_": span_id, "bound_args": bound_args, "instance": None}

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
    result = instance.func(1, 2, a=3, c=5)

    # assert
    # span_enter
    span_id = f"{instance.func.__qualname__}-mock"
    bound_args = inspect.signature(instance.func).bind(1, 2, a=3, c=5)
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == ()
    assert kwargs == {"id_": span_id, "bound_args": bound_args, "instance": instance}

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
        _ = func_exc(7, a=3, b=5, c=2, d=5)

    # assert
    # span_enter
    mock_span_enter.assert_called_once()

    # span_drop
    mock_span_drop.assert_called_once()
    span_id = f"{func_exc.__qualname__}-mock"
    bound_args = inspect.signature(func_exc).bind(7, a=3, b=5, c=2, d=5)
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
        _ = instance.func_exc(7, a=3, b=5, c=2, d=5)

    # assert
    # span_enter
    mock_span_enter.assert_called_once()

    # span_drop
    mock_span_drop.assert_called_once()
    span_id = f"{instance.func_exc.__qualname__}-mock"
    bound_args = inspect.signature(instance.func_exc).bind(7, a=3, b=5, c=2, d=5)
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
    result = await async_func(1, 2, a=3, c=5)

    # assert
    # span_enter
    span_id = f"{async_func.__qualname__}-mock"
    bound_args = inspect.signature(async_func).bind(1, 2, a=3, c=5)
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == ()
    assert kwargs == {"id_": span_id, "bound_args": bound_args, "instance": None}

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
    result = await instance.async_func(1, 2, a=3, c=5)

    # assert
    # span_enter
    span_id = f"{instance.async_func.__qualname__}-mock"
    bound_args = inspect.signature(instance.async_func).bind(1, 2, a=3, c=5)
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == ()
    assert kwargs == {"id_": span_id, "bound_args": bound_args, "instance": instance}

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
        _ = await async_func_exc(7, a=3, b=5, c=2, d=5)

    # assert
    # span_enter
    mock_span_enter.assert_called_once()

    # span_drop
    mock_span_drop.assert_called_once()
    span_id = f"{async_func_exc.__qualname__}-mock"
    bound_args = inspect.signature(async_func_exc).bind(7, a=3, b=5, c=2, d=5)
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
        _ = await instance.async_func_exc(7, a=3, b=5, c=2, d=5)

    # assert
    # span_enter
    mock_span_enter.assert_called_once()

    # span_drop
    mock_span_drop.assert_called_once()
    span_id = f"{instance.async_func_exc.__qualname__}-mock"
    bound_args = inspect.signature(instance.async_func_exc).bind(7, a=3, b=5, c=2, d=5)
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
