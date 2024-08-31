import pytest

from llama_index.core.workflow.context import Context
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy
from llama_index.core.workflow.workflow import Workflow


@pytest.mark.asyncio()
async def test_retry_e2e():
    class CountEvent(Event):
        """Empty event to signal a step to increment a counter in the Context."""

    class DummyWorkflow(Workflow):
        @step(retry_policy=ConstantDelayRetryPolicy(delay=1))
        async def flaky_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            count = await ctx.get("counter", default=0)
            ctx.send_event(CountEvent())
            if count < 3:
                raise ValueError("Something bad happened!")
            return StopEvent(result="All good!")

        @step
        async def counter(self, ctx: Context, ev: CountEvent) -> None:
            count = await ctx.get("counter", default=0)
            await ctx.set("counter", count + 1)

    workflow = DummyWorkflow(disable_validation=True)
    await workflow.run()
