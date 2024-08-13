from llama_index.core.workflow.events import Event
from llama_index.core.bridge.pydantic import PrivateAttr
from typing import Any


class _TestEvent(Event):
    param: str
    _private_param_1: str = PrivateAttr()
    _private_param_2: str = PrivateAttr(default_factory=str)


class _TestEvent2(Event):
    """Custom Test Event.

    Private Attrs:
        _private_param: doesn't get modified during construction
        _modified_private_param: gets processed before being set
    """

    _private_param: int = PrivateAttr()
    _modified_private_param: int = PrivateAttr()

    def __init__(self, _modified_private_param: int, **params: Any):
        super().__init__(**params)
        self._modified_private_param = _modified_private_param * 2


def test_event_init_basic():
    evt = Event(a=1, b=2, c="c")

    assert evt.a == 1
    assert evt.b == 2
    assert evt.c == "c"
    assert evt["a"] == evt.a
    assert evt["b"] == evt.b
    assert evt["c"] == evt.c
    assert evt.keys() == {"a": 1, "b": 2, "c": "c"}.keys()


def test_custom_event_with_fields_and_private_params():
    evt = _TestEvent(a=1, param="test_param", _private_param_1="test_private_param_1")

    assert evt.a == 1
    assert evt["a"] == evt.a
    assert evt.param == "test_param"
    assert evt._data == {"a": 1}
    assert evt._private_param_1 == "test_private_param_1"
    assert evt._private_param_2 == ""


def test_custom_event_override_init():
    evt = _TestEvent2(a=1, b=2, _private_param=2, _modified_private_param=2)

    assert evt.a == 1
    assert evt.b == 2
    assert evt._data == {"a": 1, "b": 2}
    assert evt._private_param == 2
    assert evt._modified_private_param == 4
