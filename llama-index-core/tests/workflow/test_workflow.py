import asyncio
import time
from unittest import mock

import pytest

from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent
from llama_index.core.workflow.workflow import (
    Context,
    Workflow,
    WorkflowTimeoutError,
    WorkflowValidationError,
    WorkflowRuntimeError,
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
        @step
        async def slow_step(self, ev: StartEvent) -> StopEvent:
            await asyncio.sleep(5.0)
            return StopEvent(result="Done")

    workflow = SlowWorkflow(timeout=1)
    with pytest.raises(WorkflowTimeoutError):
        await workflow.run()


@pytest.mark.asyncio()
async def test_workflow_validation():
    class InvalidWorkflow(Workflow):
        @step
        async def invalid_step(self, ev: StartEvent) -> None:
            pass

    workflow = InvalidWorkflow()
    with pytest.raises(WorkflowValidationError):
        await workflow.run()


@pytest.mark.asyncio()
async def test_workflow_event_propagation():
    events = []

    class EventTrackingWorkflow(Workflow):
        @step
        async def step1(self, ev: StartEvent) -> OneTestEvent:
            events.append("step1")
            return OneTestEvent()

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            events.append("step2")
            return StopEvent(result="Done")

    workflow = EventTrackingWorkflow()
    await workflow.run()
    assert events == ["step1", "step2"]


@pytest.mark.asyncio()
async def test_sync_async_steps():
    class SyncAsyncWorkflow(Workflow):
        @step
        async def async_step(self, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step
        def sync_step(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="Done")

    workflow = SyncAsyncWorkflow()
    await workflow.run()
    assert workflow.is_done()


@pytest.mark.asyncio()
async def test_workflow_num_workers():
    class NumWorkersWorkflow(Workflow):
        @step
        async def original_step(
            self, ctx: Context, ev: StartEvent
        ) -> OneTestEvent | LastEvent:
            ctx.data["num_to_collect"] = 3
            ctx.session.send_event(OneTestEvent(test_param="test1"))
            ctx.session.send_event(OneTestEvent(test_param="test2"))
            ctx.session.send_event(OneTestEvent(test_param="test3"))

            return LastEvent()

        @step(num_workers=3)
        async def test_step(self, ev: OneTestEvent) -> AnotherTestEvent:
            await asyncio.sleep(1.0)
            return AnotherTestEvent(another_test_param=ev.test_param)

        @step
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


@pytest.mark.asyncio()
async def test_workflow_step_send_event():
    class StepSendEventWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            ctx.session.send_event(OneTestEvent(), step="step2")
            return None

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step2")

        @step
        async def step3(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step3")

    workflow = StepSendEventWorkflow()
    result = await workflow.run()
    assert result == "step2"
    assert workflow.is_done()
    session = workflow._sessions.pop()
    assert ("step2", "OneTestEvent") in session._accepted_events
    assert ("step3", "OneTestEvent") not in session._accepted_events


@pytest.mark.asyncio()
async def test_workflow_step_send_event_to_None():
    class StepSendEventToNoneWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            ctx.session.send_event(OneTestEvent(), step=None)
            return None

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step2")

    workflow = StepSendEventToNoneWorkflow()
    await workflow.run()
    assert workflow.is_done()
    assert ("step2", "OneTestEvent") in workflow._sessions.pop()._accepted_events


@pytest.mark.asyncio()
async def test_workflow_missing_service():
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: StartEvent, my_service: Workflow) -> StopEvent:
            return StopEvent(result=42)

    workflow = DummyWorkflow()
    # do not add any service called "my_service"...
    with pytest.raises(
        WorkflowValidationError,
        match="The following services are not available: my_service",
    ):
        await workflow.run()


@pytest.mark.asyncio()
async def test_workflow_multiple_runs():
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result=ev.number * 2)

    workflow = DummyWorkflow()
    results = await asyncio.gather(
        workflow.run(number=3), workflow.run(number=42), workflow.run(number=-99)
    )
    assert set(results) == {6, 84, -198}


def test_deprecated_send_event():
    ev = StartEvent()
    wf = Workflow()
    session1 = mock.MagicMock()

    # One session, assert step emits a warning
    wf._sessions.add(session1)
    with pytest.warns(UserWarning):
        wf.send_event(message=ev)
    session1.send_event.assert_called_with(message=ev, step=None)

    # Second session, assert step raises an exception
    session2 = mock.MagicMock()
    wf._sessions.add(session2)
    with pytest.raises(WorkflowRuntimeError):
        wf.send_event(message=ev)
    session2.send_event.assert_not_called()
