import asyncio
import time

import pytest

from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent
from llama_index.core.workflow.workflow import (
    Context,
    Workflow,
    WorkflowTimeoutError,
    WorkflowValidationError,
)

from .conftest import AnotherTestEvent, LastEvent, OneTestEvent


@pytest.mark.asyncio()
async def test_workflow_initialization(workflow):
    assert workflow._timeout == 10
    assert not workflow._disable_validation
    assert not workflow._verbose


@pytest.mark.asyncio()
async def test_workflow_run(workflow):
    result = await workflow.run()
    assert result == "Workflow completed"


@pytest.mark.asyncio()
async def test_workflow_run_step(workflow):
    workflow._verbose = True

    # First step
    result = await workflow.run_step()
    assert result is None
    assert not workflow.is_done()

    # Second step
    result = await workflow.run_step()
    assert result is None
    assert not workflow.is_done()

    # Final step
    result = await workflow.run_step()
    assert not workflow.is_done()
    assert result is None

    # Cleanup step
    result = await workflow.run_step()
    assert result == "Workflow completed"
    assert workflow.is_done()


@pytest.mark.asyncio()
async def test_workflow_timeout():
    class SlowWorkflow(Workflow):
        @step()
        async def slow_step(self, ev: StartEvent) -> StopEvent:
            await asyncio.sleep(5.0)
            return StopEvent(result="Done")

    workflow = SlowWorkflow(timeout=1)
    with pytest.raises(WorkflowTimeoutError):
        await workflow.run()


@pytest.mark.asyncio()
async def test_workflow_validation():
    class InvalidWorkflow(Workflow):
        @step()
        async def invalid_step(self, ev: StartEvent) -> None:
            pass

    workflow = InvalidWorkflow()
    with pytest.raises(WorkflowValidationError):
        await workflow.run()


@pytest.mark.asyncio()
async def test_workflow_event_propagation():
    events = []

    class EventTrackingWorkflow(Workflow):
        @step()
        async def step1(self, ev: StartEvent) -> OneTestEvent:
            events.append("step1")
            return OneTestEvent()

        @step()
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            events.append("step2")
            return StopEvent(result="Done")

    workflow = EventTrackingWorkflow()
    await workflow.run()
    assert events == ["step1", "step2"]


@pytest.mark.asyncio()
async def test_sync_async_steps():
    class SyncAsyncWorkflow(Workflow):
        @step()
        async def async_step(self, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step()
        def sync_step(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="Done")

    workflow = SyncAsyncWorkflow()
    await workflow.run()
    assert workflow.is_done()


@pytest.mark.asyncio()
async def test_workflow_num_workers():
    class NumWorkersWorkflow(Workflow):
        @step(pass_context=True)
        async def original_step(
            self, ctx: Context, ev: StartEvent
        ) -> OneTestEvent | LastEvent:
            ctx.data["num_to_collect"] = 3
            self.send_event(OneTestEvent(test_param="test1"))
            self.send_event(OneTestEvent(test_param="test2"))
            self.send_event(OneTestEvent(test_param="test3"))

            return LastEvent()

        @step(num_workers=3)
        async def test_step(self, ev: OneTestEvent) -> AnotherTestEvent:
            await asyncio.sleep(1.0)
            return AnotherTestEvent(another_test_param=ev.test_param)

        @step(pass_context=True)
        async def final_step(
            self, ctx: Context, ev: AnotherTestEvent | LastEvent
        ) -> StopEvent:
            events = ctx.collect_events(
                ev, [AnotherTestEvent] * ctx.data["num_to_collect"]
            )
            if events is None:
                return None
            return StopEvent(result=[ev.another_test_param for ev in events])

    workflow = NumWorkersWorkflow()

    start_time = time.time()
    result = await workflow.run()
    end_time = time.time()

    assert workflow.is_done()
    assert set(result) == {"test1", "test2", "test3"}

    # Check if the execution time is close to 1 second (with some tolerance)
    execution_time = end_time - start_time
    assert (
        1.0 <= execution_time < 1.1
    ), f"Execution time was {execution_time:.2f} seconds"
