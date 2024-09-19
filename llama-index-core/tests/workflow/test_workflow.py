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
async def test_deprecated_workflow_run_step(workflow):
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
async def test_workflow_run_step(workflow):
    handler = workflow.run(stepwise=True)

    result = await handler.run_step()
    assert result is None
    assert not handler.is_done()

    result = await handler.run_step()
    assert result is None
    assert not handler.is_done()

    result = await handler.run_step()
    assert result is None
    assert not handler.is_done()

    result = await handler.run_step()
    assert result is None
    assert not handler.is_done()

    result = await handler.run_step()
    assert result == "Workflow completed"
    assert handler.is_done()


@pytest.mark.asyncio()
async def test_workflow_run_step_continue_context():
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            cur_number = await ctx.get("number", default=0)
            await ctx.set("number", cur_number + 1)
            return StopEvent(result="Done")


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
async def test_workflow_validation_unproduced_events():
    class InvalidWorkflow(Workflow):
        @step
        async def invalid_step(self, ev: StartEvent) -> None:
            pass

    workflow = InvalidWorkflow()
    with pytest.raises(
        WorkflowValidationError,
        match="The following events are consumed but never produced: StopEvent",
    ):
        await workflow.run()


@pytest.mark.asyncio()
async def test_workflow_validation_unconsumed_events():
    class InvalidWorkflow(Workflow):
        @step
        async def invalid_step(self, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step
        async def a_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    workflow = InvalidWorkflow()
    with pytest.raises(
        WorkflowValidationError,
        match="The following events are produced but never consumed: OneTestEvent",
    ):
        await workflow.run()


@pytest.mark.asyncio()
async def test_workflow_validation_start_event_not_consumed():
    class InvalidWorkflow(Workflow):
        @step
        async def a_step(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent()

        @step
        async def another_step(self, ev: OneTestEvent) -> OneTestEvent:
            return OneTestEvent()

    workflow = InvalidWorkflow()
    with pytest.raises(
        WorkflowValidationError,
        match="The following events are produced but never consumed: StartEvent",
    ):
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
async def test_workflow_sync_async_steps():
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
                return None  # type: ignore
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
            return None  # type: ignore

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
    ctx = workflow._contexts.pop()
    assert ("step2", "OneTestEvent") in ctx._accepted_events
    assert ("step3", "OneTestEvent") not in ctx._accepted_events


@pytest.mark.asyncio()
async def test_workflow_step_send_event_to_None():
    class StepSendEventToNoneWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            ctx.send_event(OneTestEvent(), step=None)
            return  # type:ignore

        @step
        async def step2(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step2")

    workflow = StepSendEventToNoneWorkflow(verbose=True)
    await workflow.run()
    assert workflow.is_done()
    assert ("step2", "OneTestEvent") in workflow._contexts.pop()._accepted_events


@pytest.mark.asyncio()
async def test_workflow_step_returning_bogus():
    class TestWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            return "foo"  # type:ignore

        @step
        async def step2(self, ctx: Context, ev: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step
        async def step3(self, ev: OneTestEvent) -> StopEvent:
            return StopEvent(result="step2")

    workflow = TestWorkflow()
    with pytest.warns(
        UserWarning,
        match="Step function step1 returned str instead of an Event instance.",
    ):
        await workflow.run()


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
    ctx = mock.MagicMock()

    # One context, assert step emits a warning
    wf._contexts.add(ctx)
    with pytest.warns(UserWarning):
        wf.send_event(message=ev)
    ctx.send_event.assert_called_with(message=ev, step=None)

    # Second context, assert step raises an exception
    ctx = mock.MagicMock()
    wf._contexts.add(ctx)
    with pytest.raises(WorkflowRuntimeError):
        wf.send_event(message=ev)
    ctx.send_event.assert_not_called()


def test_add_step():
    class TestWorkflow(Workflow):
        @step
        def foo_step(self, ev: StartEvent) -> None:
            pass

    with pytest.raises(
        WorkflowValidationError,
        match="A step foo_step is already part of this workflow, please choose another name.",
    ):

        @step(workflow=TestWorkflow)
        def foo_step(ev: StartEvent) -> None:
            pass


def test_add_step_not_a_step():
    class TestWorkflow(Workflow):
        @step
        def a_ste(self, ev: StartEvent) -> None:
            pass

    def another_step(ev: StartEvent) -> None:
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step function another_step is missing the `@step` decorator.",
    ):
        TestWorkflow.add_step(another_step)


@pytest.mark.asyncio()
async def test_workflow_task_raises():
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: StartEvent) -> StopEvent:
            raise ValueError("The step raised an error!")

    workflow = DummyWorkflow()
    with pytest.raises(ValueError, match="The step raised an error!"):
        await workflow.run()


@pytest.mark.asyncio()
async def test_workflow_task_raises_step():
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ev: StartEvent) -> StopEvent:
            raise ValueError("The step raised an error!")

    workflow = DummyWorkflow()
    with pytest.raises(ValueError, match="The step raised an error!"):
        await workflow.run_step()


def test_workflow_disable_validation():
    w = Workflow(disable_validation=True)
    w._get_steps = mock.MagicMock()
    w._validate()
    w._get_steps.assert_not_called()


@pytest.mark.asyncio()
async def test_workflow_continue_context():
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            cur_number = await ctx.get("number", default=0)
            await ctx.set("number", cur_number + 1)
            return StopEvent(result="Done")

    wf = DummyWorkflow()

    # first run
    r = wf.run()
    result = await r
    assert result == "Done"
    assert await r.ctx.get("number") == 1

    # second run -- independent from the first
    r = wf.run()
    result = await r
    assert result == "Done"
    assert await r.ctx.get("number") == 1

    # third run -- continue from the second run
    r = wf.run(ctx=r.ctx)
    result = await r
    assert result == "Done"
    assert await r.ctx.get("number") == 2
