import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.dispatcher import Dispatcher
from unittest.mock import patch

dispatcher = instrument.get_dispatcher("test")


@dispatcher.span
def func(*args, a, b=3, **kwargs):
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
    mock_span_exit.assert_called_once()
    args, kwargs = mock_span_exit.call_args
    assert args == (1, 2)
    assert kwargs == {"id": span_id, "a": 3, "c": 5, "result": result}
