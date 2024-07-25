from typing import Union

import pytest

from llama_index.core.workflow.errors import WorkflowValidationError
from llama_index.core.workflow.events import Event
from llama_index.core.workflow.utils import (
    validate_step_signature,
)


class TestEvent(Event):
    pass


class AnotherTestEvent(Event):
    pass


def test_validate_step_signature_of_method():
    def f(self, ev: TestEvent):
        pass

    validate_step_signature(f)


def test_validate_step_signature_of_free_function():
    def f(ev: TestEvent):
        pass

    validate_step_signature(f)


def test_validate_step_signature_union():
    def f(ev: Union[TestEvent, AnotherTestEvent]):
        pass

    validate_step_signature(f)


def test_validate_step_signature_union_invalid():
    def f(ev: Union[TestEvent, str]):
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature parameters must be annotated with an Event type",
    ):
        validate_step_signature(f)


def test_validate_step_signature_no_params():
    def f():
        pass

    with pytest.raises(
        WorkflowValidationError, match="Step signature must have at least one parameter"
    ):
        validate_step_signature(f)


def test_validate_step_signature_no_annotations():
    def f(self, ev):
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature parameters must be annotated with an Event type",
    ):
        validate_step_signature(f)


def test_validate_step_signature_wrong_annotations():
    def f(self, ev: str):
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature parameters must be annotated with an Event type",
    ):
        validate_step_signature(f)


def test_validate_step_signature_too_many_params():
    def f1(self, ev: TestEvent, foo: TestEvent):
        pass

    def f2(ev: TestEvent, foo: TestEvent):
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="must contain exactly one parameter of type Event and no other parameters",
    ):
        validate_step_signature(f1)

    with pytest.raises(
        WorkflowValidationError,
        match="must contain exactly one parameter of type Event and no other parameters",
    ):
        validate_step_signature(f2)


# def test_get_steps_from_class():
#     class Test:
#         @step()
#         def start(self, start: StartEvent) -> TestEvent:
#             pass

#         @step()
#         def my_method(self, event: TestEvent) -> StopEvent:
#             pass

#         def not_a_step(self):
#             pass

#     steps = get_steps_from_class(Test())
#     assert len(steps)
#     assert "my_method" in steps
