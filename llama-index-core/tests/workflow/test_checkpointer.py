import pytest

from llama_index.core.workflow.events import StartEvent, StopEvent
from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.core.workflow.events import (
    StartEvent,
    StopEvent,
)
from llama_index.core.workflow.workflow import (
    Context,
)
from llama_index.core.workflow.checkpointer import WorkflowCheckpointer
from .conftest import OneTestEvent, DummyWorkflow, LastEvent


@pytest.fixture()
def workflow_checkpointer(workflow: DummyWorkflow):
    return WorkflowCheckpointer(workflow=workflow)


@pytest.mark.asyncio()
async def test_create_checkpoint(workflow_checkpointer: WorkflowCheckpointer):
    incoming_ev = StartEvent()
    output_ev = OneTestEvent()
    ctx = Context(workflow=workflow_checkpointer.workflow)
    checkpointer = workflow_checkpointer.new_checkpoint_callback_for_run()

    # execute checkpoint asynccallable
    await checkpointer(
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
        handler: WorkflowHandler = workflow_checkpointer.run(store_checkpoints=True)
        await handler

    # filter by last complete step
    steps = ["start_step", "middle_step", "end_step"]  # sequential workflow
    for step in steps:
        checkpoints = workflow_checkpointer.filter_checkpoints(last_completed_step=step)
        assert len(checkpoints) == num_runs, f"fails on step: {step.__name__}"

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
async def test_checkpoints_works_with_new_instances(
    workflow_checkpointer: WorkflowCheckpointer,
):
    num_instances = 3
    for _ in range(num_instances):
        workflow = DummyWorkflow()
        workflow_checkpointer.workflow = workflow
        handler: WorkflowHandler = workflow_checkpointer.run()
        await handler

    assert len(workflow_checkpointer.checkpoints) == num_instances


@pytest.mark.asyncio()
async def test_run_from_checkpoint(workflow_checkpointer: WorkflowCheckpointer):
    print(f"{workflow_checkpointer.workflow._get_steps()}", flush=True)
    num_steps = len(workflow_checkpointer.workflow._get_steps())
    num_ckpts_in_single_run = num_steps - 1
    handler: WorkflowHandler = workflow_checkpointer.run(store_checkpoints=True)
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
    for k, v in workflow_checkpointer.checkpoints.items():
        print(f"{k}: {[c.last_completed_step for c in v]}")
    assert num_checkpoints == [1, num_ckpts_in_single_run]
