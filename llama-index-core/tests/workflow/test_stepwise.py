import asyncio

import pytest
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow


class IntermediateEvent(Event):
    value: int


class StepWorkflow(Workflow):
    @step
    async def step1(self, ev: StartEvent) -> IntermediateEvent:
        await asyncio.sleep(0.1)
        return IntermediateEvent(value=21)

    @step
    async def step2(self, ev: IntermediateEvent) -> StopEvent:
        await asyncio.sleep(0.1)
        return StopEvent(result=ev.value * 2)


@pytest.mark.asyncio()
async def test_simple_stepwise():
    workflow = StepWorkflow()
    handler = workflow.run(stepwise=True)
    while ev := await handler.run_step():
        handler.ctx.send_event(ev)  # type: ignore

    result = await handler
    assert result == 42
