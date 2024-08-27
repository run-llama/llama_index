import asyncio

import pytest

from llama_index.core.workflow.context import Context
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow
from llama_index.core.workflow.errors import WorkflowRuntimeError


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
    r = asyncio.create_task(wf.run())

    async for ev in wf.stream_events():
        assert "msg" in ev

    await r


@pytest.mark.asyncio()
async def test_too_many_runs():
    wf = StreamingWorkflow()
    r = asyncio.gather(wf.run(), wf.run())
    with pytest.raises(
        WorkflowRuntimeError,
        match="This workflow has multiple session running concurrently",
    ):
        async for ev in wf.stream_events():
            pass
