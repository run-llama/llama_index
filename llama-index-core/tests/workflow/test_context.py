import asyncio
import json
from typing import Optional, Union
from unittest import mock

import pytest
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.errors import WorkflowRuntimeError
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import (
    Context,
    Workflow,
)

from .conftest import AnotherTestEvent, OneTestEvent


@pytest.mark.asyncio()
async def test_collect_events():
    ev1 = OneTestEvent()
    ev2 = AnotherTestEvent()

    class TestWorkflow(Workflow):
        @step
        async def step1(self, _: StartEvent) -> OneTestEvent:
            return ev1

        @step
        async def step2(self, _: StartEvent) -> AnotherTestEvent:
            return ev2

        @step
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


@pytest.mark.asyncio()
async def test_get_default(workflow):
    c1 = Context(workflow)
    assert await c1.get(key="test_key", default=42) == 42


@pytest.mark.asyncio()
async def test_get(ctx):
    await ctx.set("foo", 42)
    assert await ctx.get("foo") == 42


@pytest.mark.asyncio()
async def test_get_not_found(ctx):
    with pytest.raises(ValueError):
        await ctx.get("foo")


@pytest.mark.asyncio()
async def test_legacy_data(workflow):
    c1 = Context(workflow)
    await c1.set(key="test_key", value=42)
    assert await c1.get("test_key") == 42


def test_send_event_step_is_none(ctx):
    ctx._queues = {"step1": mock.MagicMock(), "step2": mock.MagicMock()}
    ev = Event(foo="bar")
    ctx.send_event(ev)
    for q in ctx._queues.values():
        q.put_nowait.assert_called_with(ev)
    assert ctx._broker_log == [ev]


def test_send_event_to_non_existent_step(ctx):
    with pytest.raises(
        WorkflowRuntimeError, match="Step does_not_exist does not exist"
    ):
        ctx.send_event(Event(), "does_not_exist")


def test_send_event_to_wrong_step(ctx):
    ctx._workflow._get_steps = mock.MagicMock(return_value={"step": mock.MagicMock()})

    with pytest.raises(
        WorkflowRuntimeError,
        match="Step step does not accept event of type <class 'llama_index.core.workflow.events.Event'>",
    ):
        ctx.send_event(Event(), "step")


def test_send_event_to_step(ctx):
    step2 = mock.MagicMock()
    step2.__step_config.accepted_events = [Event]

    ctx._workflow._get_steps = mock.MagicMock(
        return_value={"step1": mock.MagicMock(), "step2": step2}
    )
    ctx._queues = {"step1": mock.MagicMock(), "step2": mock.MagicMock()}

    ev = Event(foo="bar")
    ctx.send_event(ev, "step2")

    ctx._queues["step1"].put_nowait.assert_not_called()
    ctx._queues["step2"].put_nowait.assert_called_with(ev)


def test_get_result(ctx):
    ctx._retval = 42
    assert ctx.get_result() == 42


def test_to_dict_with_events_buffer(ctx):
    ctx.collect_events(OneTestEvent(), [OneTestEvent, AnotherTestEvent])
    assert json.dumps(ctx.to_dict())


@pytest.mark.asyncio()
async def test_deprecated_params(ctx):
    with pytest.warns(
        DeprecationWarning, match="`make_private` is deprecated and will be ignored"
    ):
        await ctx.set("foo", 42, make_private=True)


@pytest.mark.asyncio()
async def test_empty_inprogress_when_workflow_done(workflow):
    h = workflow.run()
    _ = await h

    # there shouldn't be any in progress events
    for inprogress_list in h.ctx._in_progress.values():
        assert len(inprogress_list) == 0


@pytest.mark.asyncio()
async def test_wait_for_event(ctx):
    wait_job = asyncio.create_task(ctx.wait_for_event(Event))
    await asyncio.sleep(0.01)
    ctx.send_event(Event(msg="foo"))
    ev = await wait_job
    assert ev.msg == "foo"


@pytest.mark.asyncio()
async def test_wait_for_event_with_requirements(ctx):
    wait_job = asyncio.create_task(ctx.wait_for_event(Event, {"msg": "foo"}))
    await asyncio.sleep(0.01)
    ctx.send_event(Event(msg="bar"))
    ctx.send_event(Event(msg="foo"))
    ev = await wait_job
    assert ev.msg == "foo"


@pytest.mark.asyncio()
async def test_wait_for_event_in_workflow():
    class TestWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> StopEvent:
            ctx.write_event_to_stream(Event(msg="foo"))
            result = await ctx.wait_for_event(Event)
            return StopEvent(result=result.msg)

    workflow = TestWorkflow()
    handler = workflow.run()
    assert handler.ctx
    async for ev in handler.stream_events():
        if isinstance(ev, Event) and ev.msg == "foo":
            handler.ctx.send_event(Event(msg="bar"))
            break

    result = await handler
    assert result == "bar"
