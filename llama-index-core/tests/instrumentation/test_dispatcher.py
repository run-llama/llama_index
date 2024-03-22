import pytest
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.dispatcher import Dispatcher
from unittest.mock import patch, MagicMock

dispatcher = instrument.get_dispatcher("test")


@dispatcher.span
def func(*args, a, b=3, **kwargs):
    return a + b


@dispatcher.span
async def async_func(*args, a, b=3, **kwargs):
    return a + b


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
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == (1, 2)
    assert kwargs == {"id": span_id, "a": 3, "c": 5}

    # span_exit
    args, kwargs = mock_span_exit.call_args
    assert args == (1, 2)
    assert kwargs == {"id": span_id, "a": 3, "c": 5, "result": result}


@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
@patch(f"{__name__}.func")
def test_dispatcher_span_drop_args(
    mock_func: MagicMock,
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    class CustomException(Exception):
        pass

    mock_uuid.uuid4.return_value = "mock"
    mock_func.side_effect = CustomException

    with pytest.raises(CustomException):
        # act
        result = func(7, a=3, b=5, c=2, d=5)

        # assert
        # span_enter
        mock_span_enter.assert_called_once()

        # span_drop
        mock_span_drop.assert_called_once()
        span_id = f"{func.__qualname__}-mock"
        args, kwargs = mock_span_exit.call_args
        assert args == (7,)
        assert kwargs == {
            "id": span_id,
            "a": 3,
            "b": 5,
            "c": 2,
            "d": 2,
            "err": CustomException,
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
    mock_span_enter.assert_called_once()
    args, kwargs = mock_span_enter.call_args
    assert args == (1, 2)
    assert kwargs == {"id": span_id, "a": 3, "c": 5}

    # span_exit
    args, kwargs = mock_span_exit.call_args
    assert args == (1, 2)
    assert kwargs == {"id": span_id, "a": 3, "c": 5, "result": result}


@pytest.mark.asyncio()
@patch.object(Dispatcher, "span_exit")
@patch.object(Dispatcher, "span_drop")
@patch.object(Dispatcher, "span_enter")
@patch("llama_index.core.instrumentation.dispatcher.uuid")
@patch(f"{__name__}.async_func")
async def test_dispatcher_aysnc_span_drop_args(
    mock_func: MagicMock,
    mock_uuid: MagicMock,
    mock_span_enter: MagicMock,
    mock_span_drop: MagicMock,
    mock_span_exit: MagicMock,
):
    # arrange
    class CustomException(Exception):
        pass

    mock_uuid.uuid4.return_value = "mock"
    mock_func.side_effect = CustomException

    with pytest.raises(CustomException):
        # act
        result = await async_func(7, a=3, b=5, c=2, d=5)

        # assert
        # span_enter
        mock_span_enter.assert_called_once()

        # span_drop
        mock_span_drop.assert_called_once()
        span_id = f"{func.__qualname__}-mock"
        args, kwargs = mock_span_exit.call_args
        assert args == (7,)
        assert kwargs == {
            "id": span_id,
            "a": 3,
            "b": 5,
            "c": 2,
            "d": 2,
            "err": CustomException,
        }

        # span_exit
        mock_span_exit.assert_not_called()
