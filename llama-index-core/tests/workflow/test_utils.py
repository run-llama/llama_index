import inspect
from typing import Union, Any, Optional, List, get_type_hints

import pytest

from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.errors import WorkflowValidationError
from llama_index.core.workflow.events import StartEvent, StopEvent
from llama_index.core.workflow.utils import (
    validate_step_signature,
    inspect_signature,
    get_steps_from_class,
    get_steps_from_instance,
    _get_param_types,
    _get_return_types,
    is_free_function,
)
from llama_index.core.workflow.context import Context

from .conftest import OneTestEvent, AnotherTestEvent


def test_validate_step_signature_of_method():
    def f(self, ev: OneTestEvent) -> OneTestEvent:
        return OneTestEvent()

    validate_step_signature(inspect_signature(f))


def test_validate_step_signature_of_free_function():
    def f(ev: OneTestEvent) -> OneTestEvent:
        return OneTestEvent()

    validate_step_signature(inspect_signature(f))


def test_validate_step_signature_union():
    def f(ev: Union[OneTestEvent, AnotherTestEvent]) -> OneTestEvent:
        return OneTestEvent()

    validate_step_signature(inspect_signature(f))


def test_validate_step_signature_of_free_function_with_context():
    def f(ctx: Context, ev: OneTestEvent) -> OneTestEvent:
        return OneTestEvent()

    validate_step_signature(inspect_signature(f))


def test_validate_step_signature_union_invalid():
    def f(ev: Union[OneTestEvent, str]) -> None:
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must have at least one parameter annotated as type Event",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_no_params():
    def f() -> None:
        pass

    with pytest.raises(
        WorkflowValidationError, match="Step signature must have at least one parameter"
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_no_annotations():
    def f(self, ev) -> None:
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must have at least one parameter annotated as type Event",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_wrong_annotations():
    def f(self, ev: str) -> None:
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must have at least one parameter annotated as type Event",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_no_return_annotations():
    def f(self, ev: OneTestEvent):
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Return types of workflows step functions must be annotated with their type",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_no_events():
    def f(self, ctx: Context) -> None:
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must have at least one parameter annotated as type Event",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_too_many_params():
    def f1(self, ev: OneTestEvent, foo: OneTestEvent) -> None:
        pass

    def f2(ev: OneTestEvent, foo: OneTestEvent):
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must contain exactly one parameter of type Event but found 2.",
    ):
        validate_step_signature(inspect_signature(f1))

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must contain exactly one parameter of type Event but found 2.",
    ):
        validate_step_signature(inspect_signature(f2))


def test_get_steps_from():
    class Test:
        @step
        def start(self, start: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step
        def my_method(self, event: OneTestEvent) -> StopEvent:
            return StopEvent()

        def not_a_step(self):
            pass

    steps = get_steps_from_class(Test)
    assert len(steps)
    assert "my_method" in steps

    steps = get_steps_from_instance(Test())
    assert len(steps)
    assert "my_method" in steps


def test_get_param_types():
    def f(foo: str):
        pass

    sig = inspect.signature(f)
    type_hints = get_type_hints(f)
    res = _get_param_types(sig.parameters["foo"], type_hints)
    assert len(res) == 1
    assert res[0] is str


def test_get_param_types_no_annotations():
    def f(foo):
        pass

    sig = inspect.signature(f)
    type_hints = get_type_hints(f)
    res = _get_param_types(sig.parameters["foo"], type_hints)
    assert len(res) == 1
    assert res[0] is Any


def test_get_param_types_union():
    def f(foo: Union[str, int]):
        pass

    sig = inspect.signature(f)
    type_hints = get_type_hints(f)
    res = _get_param_types(sig.parameters["foo"], type_hints)
    assert len(res) == 2
    assert res == [str, int]


def test_get_return_types():
    def f(foo: int) -> str:
        return ""

    assert _get_return_types(f) == [str]


def test_get_return_types_union():
    def f(foo: int) -> Union[str, int]:
        return ""

    assert _get_return_types(f) == [str, int]


def test_get_return_types_optional():
    def f(foo: int) -> Optional[str]:
        return ""

    assert _get_return_types(f) == [str]


def test_get_return_types_list():
    def f(foo: int) -> List[str]:
        return [""]

    assert _get_return_types(f) == [List[str]]


def test_is_free_function():
    assert is_free_function("my_function") is True
    assert is_free_function("MyClass.my_method") is False
    assert is_free_function("some_function.<locals>.my_function") is True
    assert is_free_function("some_function.<locals>.MyClass.my_function") is False
    with pytest.raises(ValueError):
        is_free_function("")
