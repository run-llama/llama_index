import pickle
import threading
from typing import Any, Callable

import pytest
from llama_index.core.schema import BaseComponent
from pydantic.fields import PrivateAttr


class _PickleTestComponent(BaseComponent):
    fn: Any = None
    name: str = "ok"
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)


@pytest.fixture()
def my_component():
    class MyComponent(BaseComponent):
        foo: str = "bar"

    return MyComponent


def test_identifiers():
    assert BaseComponent.class_name() == "base_component"


def test_schema():
    assert BaseComponent.model_json_schema() == {
        "description": "Base component object to capture class names.",
        "properties": {
            "class_name": {
                "default": "base_component",
                "title": "Class Name",
                "type": "string",
            }
        },
        "title": "BaseComponent",
        "type": "object",
    }


def test_json():
    assert BaseComponent().json() == '{"class_name": "base_component"}'


def test__getstate__():
    class MyComponent(BaseComponent):
        _text: str = PrivateAttr(default="test private attr")
        _fn: Callable = PrivateAttr(default=lambda x: x)

    mc = MyComponent()
    # add an unpickable field
    mc._unpickable = lambda x: x  # type: ignore
    assert mc.__getstate__() == {
        "__dict__": {},
        "__pydantic_extra__": None,
        "__pydantic_fields_set__": set(),
        "__pydantic_private__": {"_text": "test private attr"},
    }
    assert callable(mc._unpickable)
    assert callable(mc._fn)


def test_pickle_preserves_unpickleable_attributes_on_live_instance():
    mc = _PickleTestComponent(fn=lambda x: x * 2)
    original_lock = mc._lock
    payload = pickle.dumps(mc)
    restored = pickle.loads(payload)

    assert restored.name == "ok"
    assert restored.fn is None
    assert mc.fn(3) == 6
    # Guard for future Pydantic versions; current BaseModel.__getstate__ already copies __pydantic_private__.
    assert mc._lock is original_lock
    # Repeated pickling should continue working without corrupting the live instance.
    assert pickle.loads(pickle.dumps(mc)).name == "ok"
    assert mc.fn(4) == 8


def test__setstate__():
    c = BaseComponent()
    c.__setstate__({})


def test_from_dict(my_component):
    mc = my_component.from_dict(
        {"class_name": "to_be_popped_out", "foo": "test string"}
    )
    assert mc.foo == "test string"


def test_from_json(my_component):
    mc = my_component.from_json(
        '{"class_name": "to_be_popped_out", "foo": "test string"}'
    )
    assert mc.foo == "test string"
