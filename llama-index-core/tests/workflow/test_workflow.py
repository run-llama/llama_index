import pytest
import asyncio
from unittest.mock import patch

from llama_index.core.workflow.workflow import (
    Workflow,
    WorkflowValidationError,
    WorkflowTimeoutError,
)
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StartEvent, StopEvent, Event


class TestEvent(Event):
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


@pytest.mark.asyncio()
async def test_workflow_initialization():
    workflow = DummyWorkflow()
    assert workflow._timeout == 10
    assert not workflow._disable_validation
    assert not workflow._verbose


@pytest.mark.asyncio()
async def test_workflow_run():
    workflow = DummyWorkflow()
    result = await workflow.run()
    assert result == "Workflow completed"


@pytest.mark.asyncio()
async def test_workflow_run_step():
    workflow = DummyWorkflow(verbose=True)

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
            return StopEvent(msg="Done")

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
            return StopEvent(msg="Done")

    workflow = EventTrackingWorkflow()
    await workflow.run()
    assert events == ["step1", "step2"]


@pytest.mark.asyncio()
async def test_workflow_draw_methods():
    workflow = DummyWorkflow()
    with patch("pyvis.network.Network") as mock_network:
        workflow.draw_all_possible_flows(filename="test_all_flows.html")
        mock_network.assert_called_once()
        mock_network.return_value.show.assert_called_once_with(
            "test_all_flows.html", notebook=False
        )

    await workflow.run()
    with patch("pyvis.network.Network") as mock_network:
        workflow.draw_most_recent_execution(filename="test_recent_execution.html")
        mock_network.assert_called_once()
        mock_network.return_value.show.assert_called_once_with(
            "test_recent_execution.html", notebook=False
        )
