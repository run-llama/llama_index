from typing import Any, Callable

import pytest
from llama_index.core.schema import BaseComponent
from pydantic.fields import PrivateAttr


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


def test__getstate__does_not_mutate_live_object():
    """
    __getstate__ must not strip attributes from the live instance.

    Regression for GitHub issue #22578: __getstate__ was deleting from the
    live __dict__ / __pydantic_private__ instead of from a copy, so the
    original instance became corrupt after the call.
    """

    class MyComponent(BaseComponent):
        fn: Any = None
        _private_fn: Callable = PrivateAttr(default=lambda x: x)

    unpickleable = lambda x: x  # noqa: E731
    mc = MyComponent(fn=unpickleable)

    # __getstate__ is what pickle calls; invoke it directly so we can verify
    # the live object is unaffected even when the class itself can't be pickled.
    state = mc.__getstate__()

    # The serialised snapshot must NOT contain the unpickleable field.
    assert "fn" not in state["__dict__"], (
        "Expected unpickleable 'fn' to be stripped from the pickle snapshot"
    )
    assert "_private_fn" not in (state.get("__pydantic_private__") or {}), (
        "Expected unpickleable '_private_fn' to be stripped from the pickle snapshot"
    )

    # The *live* instance must still have its attributes intact.
    assert mc.fn is unpickleable, (
        "__getstate__ mutated the live object and removed mc.fn"
    )
    assert callable(mc._private_fn), (
        "__getstate__ mutated the live object and removed mc._private_fn"
    )


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
