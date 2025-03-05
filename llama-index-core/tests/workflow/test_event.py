from typing import Any, cast

import pytest
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.workflow.context_serializers import JsonSerializer
from llama_index.core.workflow.events import Event


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
    evt = _TestEvent(a=1, param="test_param", _private_param_1="test_private_param_1")  # type: ignore

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


def test_event_missing_key():
    ev = _TestEvent(param="bar")
    with pytest.raises(AttributeError):
        ev.wrong_key


def test_event_not_a_field():
    ev = _TestEvent(param="foo", not_a_field="bar")  # type: ignore
    assert ev._data["not_a_field"] == "bar"
    ev.not_a_field = "baz"
    assert ev._data["not_a_field"] == "baz"
    ev["not_a_field"] = "barbaz"
    assert ev._data["not_a_field"] == "barbaz"
    assert ev.get("not_a_field") == "barbaz"


def test_event_dict_api():
    ev = _TestEvent(param="foo")
    assert len(ev) == 0
    ev["a_new_key"] = "bar"
    assert len(ev) == 1
    assert list(ev.values()) == ["bar"]
    k, v = next(iter(ev.items()))
    assert k == "a_new_key"
    assert v == "bar"
    assert next(iter(ev)) == "a_new_key"
    assert ev.dict() == {"a_new_key": "bar"}


def test_event_serialization():
    ev = _TestEvent(param="foo", not_a_field="bar")  # type: ignore
    serializer = JsonSerializer()
    serialized_ev = serializer.serialize(ev)
    deseriazlied_ev = serializer.deserialize(serialized_ev)

    assert type(deseriazlied_ev).__name__ == type(ev).__name__
    deseriazlied_ev = cast(
        _TestEvent,
        deseriazlied_ev,
    )
    assert ev.param == deseriazlied_ev.param
    assert ev._data == deseriazlied_ev._data
