from typing import Callable

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
