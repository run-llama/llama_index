import asyncio
import pytest

from llama_index.core.workflow.context import Context
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow
from llama_index.core.workflow.errors import WorkflowRuntimeError, WorkflowTimeoutError

from .conftest import OneTestEvent


class StreamingWorkflow(Workflow):
    @step
    async def chat(self, ctx: Context, ev: StartEvent) -> StopEvent:
        async def stream_messages():
            resp = "Paul Graham is a British-American computer scientist, entrepreneur, vc, and writer."
            for word in resp.split():
                yield word

        async for w in stream_messages():
            ctx.session.write_event_to_stream(Event(msg=w))

        return StopEvent(result=None)


@pytest.mark.asyncio()
async def test_e2e():
    wf = StreamingWorkflow()
    r = wf.run()

    async for ev in r.stream_events():
        if not isinstance(ev, StopEvent):
            assert "msg" in ev

    await r


@pytest.mark.asyncio()
async def test_too_many_runs():
    wf = StreamingWorkflow()
    r = asyncio.gather(wf.run(), wf.run())
    with pytest.raises(
        WorkflowRuntimeError,
        match="This workflow has multiple concurrent runs in progress and cannot stream events",
    ):
        async for ev in wf.stream_events():
            pass
    await r


@pytest.mark.asyncio()
async def test_task_raised():
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            ctx.write_event_to_stream(OneTestEvent(test_param="foo"))
            raise ValueError("The step raised an error!")

    wf = DummyWorkflow()
    r = wf.run()

    # Make sure we don't block indefinitely here because the step raised
    async for ev in r.stream_events():
        if not isinstance(ev, StopEvent):
            assert ev.test_param == "foo"

    # Make sure the await actually caught the exception
    with pytest.raises(
        WorkflowRuntimeError, match="Error in step 'step': The step raised an error!"
    ):
        await r


@pytest.mark.asyncio()
async def test_task_timeout():
    class DummyWorkflow(Workflow):
        @step
        async def step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            ctx.write_event_to_stream(OneTestEvent(test_param="foo"))
            await asyncio.sleep(2)
            return StopEvent()

    wf = DummyWorkflow(timeout=1)
    r = wf.run()

    # Make sure we don't block indefinitely here because the step raised
    async for ev in r.stream_events():
        if not isinstance(ev, StopEvent):
            assert ev.test_param == "foo"

    # Make sure the await actually caught the exception
    with pytest.raises(WorkflowTimeoutError, match="Operation timed out"):
        await r


@pytest.mark.asyncio()
async def test_multiple_sequential_streams():
    wf = StreamingWorkflow()
    r = wf.run()

    # stream 1
    async for _ in r.stream_events():
        pass
    await r

    # stream 2 -- should not raise an error
    r = wf.run()
    async for _ in r.stream_events():
        pass
    await r


@pytest.mark.asyncio()
async def test_multiple_ongoing_streams():
    wf = StreamingWorkflow()
    stream_1 = wf.run()
    stream_2 = wf.run()

    async for ev in stream_1.stream_events():
        if not isinstance(ev, StopEvent):
            assert "msg" in ev

    async for ev in stream_2.stream_events():
        if not isinstance(ev, StopEvent):
            assert "msg" in ev


@pytest.mark.asyncio()
async def test_resume_streams():
    class CounterWorkflow(Workflow):
        @step
        async def count(self, ctx: Context, ev: StartEvent) -> StopEvent:
            ctx.write_event_to_stream(Event(msg="hello!"))

            cur_count = await ctx.get("cur_count", default=0)
            await ctx.set("cur_count", cur_count + 1)
            return StopEvent(result="done")

    wf = CounterWorkflow()
    handler_1 = wf.run()

    async for _ in handler_1.stream_events():
        pass
    await handler_1

    handler_2 = wf.run(ctx=handler_1.ctx)
    async for _ in handler_2.stream_events():
        pass
    await handler_2

    assert await handler_2.ctx.get("cur_count") == 2
