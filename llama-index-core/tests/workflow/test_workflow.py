import pytest
import asyncio

from llama_index.core.workflow.workflow import (
    Workflow,
    WorkflowValidationError,
    WorkflowTimeoutError,
)
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent


from .conftest import TestEvent


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
        async def step1(self, ev: StartEvent) -> TestEvent:
            events.append("step1")
            return TestEvent()

        @step()
        async def step2(self, ev: TestEvent) -> StopEvent:
            events.append("step2")
            return StopEvent(result="Done")

    workflow = EventTrackingWorkflow()
    await workflow.run()
    assert events == ["step1", "step2"]


@pytest.mark.asyncio()
async def test_sync_async_steps():
    class SyncAsyncWorkflow(Workflow):
        @step()
        async def async_step(self, ev: StartEvent) -> TestEvent:
            return TestEvent()

        @step()
        def sync_step(self, ev: TestEvent) -> StopEvent:
            return StopEvent(result="Done")

    workflow = SyncAsyncWorkflow()
    await workflow.run()
    assert workflow.is_done()
