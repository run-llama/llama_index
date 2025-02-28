import asyncio
import random
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.workflow.checkpointer import WorkflowCheckpointer
from llama_index.core.workflow.events import (
    StartEvent,
    StopEvent,
)
from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.core.workflow.workflow import (
    Context,
)

from .conftest import DummyWorkflow, LastEvent, OneTestEvent


@pytest.fixture()
def workflow_checkpointer(workflow: DummyWorkflow):
    return WorkflowCheckpointer(workflow=workflow)


@pytest.mark.asyncio()
async def test_create_checkpoint(workflow_checkpointer: WorkflowCheckpointer):
    incoming_ev = StartEvent()
    output_ev = OneTestEvent()
    ctx = Context(workflow=workflow_checkpointer.workflow)
    checkpointer = workflow_checkpointer.new_checkpoint_callback_for_run()

    # execute checkpoint async callable
    await checkpointer(
        run_id="42",
        last_completed_step="start_step",
        input_ev=incoming_ev,
        output_ev=output_ev,
        ctx=ctx,
    )

    ckpts = next(iter(workflow_checkpointer.checkpoints.values()))
    ckpt = ckpts[0]
    assert ckpt.input_event == incoming_ev
    assert ckpt.output_event == output_ev
    assert ckpt.last_completed_step == "start_step"
    # should be the same since nothing happened between snapshot and creating ckpt
    assert ckpt.ctx_state == ctx.to_dict()


@pytest.mark.asyncio()
async def test_checkpoints_after_successive_runs(
    workflow_checkpointer: WorkflowCheckpointer,
):
    num_steps = len(workflow_checkpointer.workflow._get_steps())
    num_runs = 2

    for _ in range(num_runs):
        handler: WorkflowHandler = workflow_checkpointer.run()
        await handler

    assert len(workflow_checkpointer.checkpoints) == num_runs
    for ckpts in workflow_checkpointer.checkpoints.values():
        assert len(ckpts) == num_steps - 1  # don't checkpoint after _done step
        assert [ckpt.last_completed_step for ckpt in ckpts] == [
            "start_step",
            "middle_step",
            "end_step",
        ]


@pytest.mark.asyncio()
async def test_filter_checkpoints(workflow_checkpointer: WorkflowCheckpointer):
    num_runs = 2
    for _ in range(num_runs):
        handler: WorkflowHandler = workflow_checkpointer.run()
        await handler

    # filter by last complete step
    steps = ["start_step", "middle_step", "end_step"]  # sequential workflow
    for step in steps:
        checkpoints = workflow_checkpointer.filter_checkpoints(last_completed_step=step)
        assert len(checkpoints) == num_runs, f"fails on step: {step}"

    # filter by input and output event
    event_types = [StartEvent, OneTestEvent, LastEvent, StopEvent]
    for evt_type in event_types:
        # by input_event_type
        if evt_type != StopEvent:
            checkpoints_by_input_event = workflow_checkpointer.filter_checkpoints(
                input_event_type=evt_type
            )
            assert (
                len(checkpoints_by_input_event) == num_runs
            ), f"fails on {evt_type.__name__}"

        # by output_event_type
        if evt_type != StartEvent:
            checkpoints_by_output_event = workflow_checkpointer.filter_checkpoints(
                output_event_type=evt_type
            )
            assert len(checkpoints_by_output_event) == num_runs, f"fails on {evt_type}"

    # no filters raises error
    with pytest.raises(ValueError):
        workflow_checkpointer.filter_checkpoints()


@pytest.mark.asyncio()
async def test_checkpoints_works_with_new_instances_concurrently(
    workflow_checkpointer: WorkflowCheckpointer,
):
    num_instances = 3
    tasks = []

    async def add_random_startup(coro: WorkflowHandler):
        """To randomly mix up the processing of the 3 runs."""
        startup = random.random()
        await asyncio.sleep(startup)
        await coro

    for _ in range(num_instances):
        workflow = DummyWorkflow()
        workflow_checkpointer.workflow = workflow
        task = asyncio.create_task(add_random_startup(workflow_checkpointer.run()))
        tasks.append(task)

    await asyncio.gather(*tasks)

    assert len(workflow_checkpointer.checkpoints) == num_instances
    for ckpts in workflow_checkpointer.checkpoints.values():
        assert [c.last_completed_step for c in ckpts] == [
            "start_step",
            "middle_step",
            "end_step",
        ]


@pytest.mark.asyncio()
async def test_run_from_checkpoint(workflow_checkpointer: WorkflowCheckpointer):
    num_steps = len(workflow_checkpointer.workflow._get_steps())
    num_ckpts_in_single_run = num_steps - 1
    handler: WorkflowHandler = workflow_checkpointer.run()
    await handler

    assert len(workflow_checkpointer.checkpoints) == 1

    # get the checkpoint after middle_step completed
    ckpt = workflow_checkpointer.filter_checkpoints(last_completed_step="middle_step")[
        0
    ]
    handler: WorkflowHandler = workflow_checkpointer.run_from(checkpoint=ckpt)
    await handler

    assert len(workflow_checkpointer.checkpoints) == 2

    # full run should have 2 checkpoints, where as the run_from call should only store 1 checkpoint
    num_checkpoints = []
    for ckpts in workflow_checkpointer.checkpoints.values():
        num_checkpoints.append(len(ckpts))
    num_checkpoints = sorted(num_checkpoints)
    assert num_checkpoints == [1, num_ckpts_in_single_run]


@pytest.mark.asyncio()
@patch("llama_index.core.workflow.workflow.uuid")
async def test_checkpointer_with_stepwise(
    mock_uuid: MagicMock,
    workflow_checkpointer: WorkflowCheckpointer,
):
    # -------------------------------
    # Stepwise run with checkpointing
    stepwise_run_id = "stepwise_run"
    mock_uuid.uuid4.return_value = stepwise_run_id
    handler = workflow_checkpointer.run(stepwise=True)
    assert handler.ctx

    event = await handler.run_step()
    assert event
    assert len(workflow_checkpointer.checkpoints[stepwise_run_id]) == 1
    handler.ctx.send_event(event)

    event = await handler.run_step()
    assert event
    assert len(workflow_checkpointer.checkpoints[stepwise_run_id]) == 2
    handler.ctx.send_event(event)

    event = await handler.run_step()
    assert event
    assert len(workflow_checkpointer.checkpoints[stepwise_run_id]) == 3
    handler.ctx.send_event(event)

    _ = await handler.run_step()
    result = await handler
    assert result == "Workflow completed"
    # -------------------------------

    # -------------------------------
    # Regular run but from a saved checkpoint from the previous stepwise run
    regular_run_id = "regular_run_id"
    ckpt = workflow_checkpointer.filter_checkpoints(last_completed_step="middle_step")[
        0
    ]
    mock_uuid.uuid4.return_value = regular_run_id
    handler = workflow_checkpointer.run_from(checkpoint=ckpt)
    result = await handler
    assert result == "Workflow completed"
    # -------------------------------

    assert [
        c.last_completed_step
        for c in workflow_checkpointer.checkpoints[stepwise_run_id]
    ] == ["start_step", "middle_step", "end_step"]

    assert [
        c.last_completed_step for c in workflow_checkpointer.checkpoints[regular_run_id]
    ] == ["end_step"]


@pytest.mark.asyncio()
@patch("llama_index.core.workflow.workflow.uuid")
async def test_disable_and_enable_checkpoints(
    mock_uuid: MagicMock,
    workflow_checkpointer: WorkflowCheckpointer,
):
    run_ids = ["42", "84"]
    mock_uuid.uuid4.side_effect = run_ids

    # disable checkpoint at middle_step
    workflow_checkpointer.disable_checkpoint("middle_step")
    handler = workflow_checkpointer.run()
    await handler

    # enable checkpoint at middle_step
    workflow_checkpointer.enable_checkpoint("middle_step")
    handler = workflow_checkpointer.run()
    await handler

    assert [
        c.last_completed_step for c in workflow_checkpointer.checkpoints[run_ids[0]]
    ] == ["start_step", "end_step"]
    assert [
        c.last_completed_step for c in workflow_checkpointer.checkpoints[run_ids[1]]
    ] == ["start_step", "middle_step", "end_step"]
