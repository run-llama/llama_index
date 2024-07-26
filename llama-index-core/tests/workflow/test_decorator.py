import re

import pytest

from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event
from llama_index.core.workflow.errors import WorkflowValidationError


def test_decorated_config(workflow):
    def f(self, ev: Event) -> Event:
        return Event()

    res = step(workflow=workflow)(f)
    config = getattr(res, "__step_config")
    assert config.accepted_events == [Event]
    assert config.event_name == "ev"
    assert config.return_types == [Event]


def test_decorate_wrong_signature():
    def f():
        pass

    with pytest.raises(WorkflowValidationError):
        step()(f)


def test_decorate_free_function(workflow):
    @step(workflow=workflow)
    def f(ev: Event) -> Event:
        return Event()

    assert workflow._step_functions == {"f": f}


def test_decorate_free_function_wrong_decorator():
    with pytest.raises(
        WorkflowValidationError,
        match=re.escape(
            "To decorate f please pass a workflow instance to the @step() decorator."
        ),
    ):

        @step()
        def f(ev: Event) -> Event:
            return Event()
