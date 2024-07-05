from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event
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


def test_valid_step_signature_ko():
    def empty():
        pass

    def toomany(self, a, b):
        pass

    def kwargs_only(self, *, event):
        pass

    assert not valid_step_signature(empty)
    assert not valid_step_signature(toomany)
    assert not valid_step_signature(kwargs_only)


def test_get_steps_from_class():
    class Test:
        @step(TestEvent)
        def my_method(self, *args):
            pass

        def not_a_step(self):
            pass

    steps = get_steps_from_class(Test())
    assert len(steps)
    assert "my_method" in steps
