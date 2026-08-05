import pickle
from typing import Any, Callable

import pytest
from llama_index.core.schema import BaseComponent
from pydantic.fields import PrivateAttr


class PickleableComponent(BaseComponent):
    """Defined at module level so that instances can actually be pickled."""

    fn: Any = None


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


def test__getstate__does_not_mutate_the_original():
    """Pickling must not strip unpickleable attributes off the live object."""

    class MyComponent(BaseComponent):
        fn: Any = None
        _private_fn: Any = PrivateAttr(default=None)

    mc = MyComponent(fn=lambda x: x)
    mc._private_fn = lambda x: x

    state = mc.__getstate__()

    # the emitted state still drops what cannot be pickled ...
    assert "fn" not in state["__dict__"]
    assert "_private_fn" not in state["__pydantic_private__"]

    # ... but the object that was pickled is left intact
    assert mc.fn is not None
    assert mc._private_fn is not None


def test_pickle_round_trip_leaves_original_usable():
    """A real `pickle.dumps` must not break the object it serialized."""
    component = PickleableComponent(fn=lambda x: x)

    restored = pickle.loads(pickle.dumps(component))

    # the unpickleable attribute is absent from the copy ...
    assert restored.fn is None
    # ... and still present on the original
    assert component.fn is not None
    assert component.fn(1) == 1


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
