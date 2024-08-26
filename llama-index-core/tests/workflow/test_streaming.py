import asyncio

import pytest

from llama_index.llms.openai import OpenAI

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow.context import Context
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow


class StreamingWorkflow(Workflow):
    @step
    async def chat(self, ctx: Context, ev: StartEvent) -> StopEvent:
        llm = OpenAI()
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a pirate with a colorful personality",
            ),
            ChatMessage(role=MessageRole.USER, content="Who is Paul Graham?"),
        ]
        gen = await llm.astream_chat(messages)
        async for resp in gen:
            ctx.session.write_stream_event(Event(msg=resp.delta))

        return StopEvent(result=None)


@pytest.mark.asyncio()
async def test_foo():
    wf = StreamingWorkflow()
    r = asyncio.create_task(wf.run())

    async for ev in wf.stream_events():
        print(ev.msg)

    await r
