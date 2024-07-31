import pytest
from typing import Union, Optional

from llama_index.core.workflow.workflow import (
    Workflow,
    Context,
)
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent

from .conftest import OneTestEvent, AnotherTestEvent


@pytest.mark.asyncio()
async def test_collect_events():
    ev1 = OneTestEvent()
    ev2 = AnotherTestEvent()

    class TestWorkflow(Workflow):
        @step()
        async def step1(self, _: StartEvent) -> OneTestEvent:
            return ev1

        @step()
        async def step2(self, _: StartEvent) -> AnotherTestEvent:
            return ev2

        @step(pass_context=True)
        async def step3(
            self, ctx: Context, ev: Union[OneTestEvent, AnotherTestEvent]
        ) -> Optional[StopEvent]:
            events = ctx.collect_events(ev, [OneTestEvent, AnotherTestEvent])
            if events is None:
                return None
            return StopEvent(result=events)

    workflow = TestWorkflow()
    result = await workflow.run()
    assert result == [ev1, ev2]
