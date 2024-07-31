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
async def test_collect_params():
    class TestWorkflow(Workflow):
        @step()
        async def step1(self, _: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step()
        async def step2(self, _: StartEvent) -> AnotherTestEvent:
            return AnotherTestEvent()

        @step(pass_context=True)
        async def step3(
            self, ctx: Context, ev: Union[OneTestEvent, AnotherTestEvent]
        ) -> Optional[StopEvent]:
            params = ctx.collect_params(ev, "test_param", "another_test_param")
            if params is None:
                return None
            return StopEvent(result=params)

    workflow = TestWorkflow()
    result = await workflow.run()
    assert result == {"test_param": "test", "another_test_param": "another_test"}
