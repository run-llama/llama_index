import pytest

from llama_index.core.workflow.events import Event


@pytest.fixture()
def events():
    class TestEvent(Event):
        pass

    class AnotherTestEvent(Event):
        pass

    return [TestEvent, AnotherTestEvent]
