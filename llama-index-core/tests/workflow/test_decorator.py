import pytest

from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event
from llama_index.core.workflow.errors import WorkflowValidationError


def test_decorated_config():
    def f(self, ev: Event) -> Event:
        return Event()

    res = step()(f)
    config = getattr(res, "__step_config")
    assert config.accepted_events == [Event]
    assert config.event_name == "ev"
    assert config.return_types == [Event]


def test_decorate_wrong_signature():
    def f():
        pass

    with pytest.raises(WorkflowValidationError):
        step()(f)
