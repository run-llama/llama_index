import pytest

from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow


class ServiceWorkflow(Workflow):
    @step()
    async def generate(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=42)


class NumGenerated(Event):
    num: int


class DummyWorkflow(Workflow):
    @step(services=["service_workflow"])
    async def get_a_number(self, svc: Workflow, ev: StartEvent) -> NumGenerated:
        res = await svc.run()
        return NumGenerated(num=int(res))

    @step()
    async def multiply(self, ev: NumGenerated) -> StopEvent:
        return StopEvent(ev.num * 2)


@pytest.mark.asyncio()
async def test_e2e():
    wf = DummyWorkflow()
    wf.add_service("service_workflow", ServiceWorkflow())
    res = await wf.run()
    print(res)
