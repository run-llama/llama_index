from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.utils import valid_step_signature, get_steps_from_class


class TestEvent(Event):
    pass


class AnotherTestEvent(Event):
    pass


def test_valid_step_signature():
    def args(*args):
        pass

    def method(self, *args):
        pass

    def arg(self, event):
        pass

    assert valid_step_signature(args)
    assert valid_step_signature(method)
    assert valid_step_signature(arg)


def test_get_steps_from_class():
    class Test:
        @step()
        def start(self, start: StartEvent) -> TestEvent:
            pass

        @step()
        def my_method(self, event: TestEvent) -> StopEvent:
            pass

        def not_a_step(self):
            pass

    steps = get_steps_from_class(Test())
    assert len(steps)
    assert "my_method" in steps
