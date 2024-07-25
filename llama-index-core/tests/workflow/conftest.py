import pytest

from llama_index.core.workflow.workflow import Workflow
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent, Event


class TestEvent(Event):
    pass


class AnotherTestEvent(Event):
    pass


class LastEvent(Event):
    pass


class DummyWorkflow(Workflow):
    @step()
    async def start_step(self, ev: StartEvent) -> TestEvent:
        return TestEvent()

    @step()
    async def middle_step(self, ev: TestEvent) -> LastEvent:
        return LastEvent()

    @step()
    async def end_step(self, ev: LastEvent) -> StopEvent:
        return StopEvent(msg="Workflow completed")


@pytest.fixture()
def workflow():
    return DummyWorkflow()


@pytest.fixture()
def events():
    return [TestEvent, AnotherTestEvent]
