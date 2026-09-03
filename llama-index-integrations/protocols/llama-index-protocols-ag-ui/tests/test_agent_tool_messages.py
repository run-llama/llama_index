import asyncio

from llama_index.core.llms import ChatMessage
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import ToolOutput
from llama_index.core.workflow.events import StopEvent
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow, ToolCallResultEvent


def backend_calc() -> str:
    return "5"


def frontend_show() -> str:
    return ""


class FakeStore:
    def __init__(self, values):
        self.values = values

    async def get(self, key, default=None):
        return self.values.get(key, default)

    async def set(self, key, value):
        self.values[key] = value


class FakeContext:
    def __init__(self, tool_results):
        self.tool_results = tool_results
        self.store = FakeStore(
            {
                "num_tool_calls": len(tool_results),
                "chat_history": [
                    ChatMessage(role="user", content="Compute 2+3 and show it."),
                    ChatMessage(role="assistant", content=""),
                ],
            }
        )
        self.stream_events = []

    def collect_events(self, ev, expected):
        return self.tool_results

    def write_event_to_stream(self, event):
        self.stream_events.append(event)


def test_aggregate_tool_calls_persists_frontend_tool_messages():
    backend_result = ToolCallResultEvent(
        tool_call_id="call_backend",
        tool_name="backend_calc",
        tool_kwargs={},
        tool_output=ToolOutput(
            tool_name="backend_calc",
            content="5",
            raw_input={},
            raw_output="5",
        ),
    )
    frontend_result = ToolCallResultEvent(
        tool_call_id="call_frontend",
        tool_name="frontend_show",
        tool_kwargs={},
        tool_output=ToolOutput(
            tool_name="frontend_show",
            content="",
            raw_input={},
            raw_output="",
        ),
    )
    workflow = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(),
        backend_tools=[backend_calc],
        frontend_tools=[frontend_show],
    )
    ctx = FakeContext([backend_result, frontend_result])

    result = asyncio.run(workflow.aggregate_tool_calls(ctx, backend_result))

    assert isinstance(result, StopEvent)
    chat_history = ctx.store.values["chat_history"]
    tool_call_ids = [
        msg.additional_kwargs["tool_call_id"]
        for msg in chat_history
        if msg.role.value == "tool"
    ]
    assert tool_call_ids == ["call_backend", "call_frontend"]
