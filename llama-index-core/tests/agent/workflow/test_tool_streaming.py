import pytest

from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool


async def streaming_tool_async():
    yield {"status": "loading", "is_preliminary": True}
    yield {"status": "done", "is_preliminary": False}


def streaming_tool_sync():
    yield {"status": "loading", "is_preliminary": True}
    yield {"status": "done", "is_preliminary": False}


def _response_generator_from_list(responses):
    index = 0

    def generator(messages):
        nonlocal index
        if not responses:
            return ChatMessage(role=MessageRole.ASSISTANT, content=None)
        msg = responses[index]
        index = min(index + 1, len(responses) - 1)
        return msg

    return generator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_fn", "tool_name", "use_async"),
    [
        (streaming_tool_async, "streaming_tool_async", True),
        (streaming_tool_sync, "streaming_tool_sync", False),
    ],
)
async def test_streaming_tool_results(tool_fn, tool_name, use_async):
    tool_call_msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=None,
        additional_kwargs={
            "tool_calls": [
                ToolSelection(tool_id="tool-1", tool_name=tool_name, tool_kwargs={})
            ]
        },
    )
    final_msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="All done.",
    )

    llm = MockFunctionCallingLLM(
        response_generator=_response_generator_from_list([tool_call_msg, final_msg])
    )

    tool = (
        FunctionTool.from_defaults(async_fn=tool_fn)
        if use_async
        else FunctionTool.from_defaults(fn=tool_fn)
    )
    agent = FunctionAgent(
        name="streaming_agent",
        description="Test agent for streaming tool outputs",
        tools=[tool],
        llm=llm,
    )

    handler = agent.run(user_msg="run the tool")
    stream_events = []
    async for ev in handler.stream_events():
        stream_events.append(ev)

    await handler

    tool_events = [ev for ev in stream_events if isinstance(ev, ToolCallResult)]
    assert any(ev.preliminary for ev in tool_events)
    assert any(not ev.preliminary for ev in tool_events)
    assert any(
        (not ev.preliminary)
        and isinstance(ev.tool_output.raw_output, dict)
        and ev.tool_output.raw_output.get("status") == "done"
        for ev in tool_events
    )
