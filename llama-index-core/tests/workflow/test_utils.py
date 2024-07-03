from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event
from llama_index.core.workflow.utils import (
    get_events_from_signature,
    get_steps_from_class,
)


class TestEvent(Event):
    pass


class AnotherTestEvent(Event):
    pass


def test_get_events_from_signature():
    def func(ev1: TestEvent, ev2: AnotherTestEvent):
        pass

    assert get_events_from_signature(func) == [TestEvent, AnotherTestEvent]


def test_get_events_from_signature_empty():
    def func():
        pass

    assert get_events_from_signature(func) == []


def test_get_events_from_signature_mixed_params():
    def func(self, some_ev: AnotherTestEvent, another_param: str):
        pass

    assert get_events_from_signature(func) == [AnotherTestEvent]


def test_get_steps_from_class():
    class Test:
        @step
        def my_method(self, ev: TestEvent):
            pass

        def not_a_step(self):
            pass

    steps = get_steps_from_class(Test())
    assert len(steps)
    assert "my_method" in steps
