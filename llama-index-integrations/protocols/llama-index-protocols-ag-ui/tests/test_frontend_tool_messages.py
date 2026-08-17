import asyncio

from ag_ui.core import RunAgentInput, UserMessage

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow


async def backend_calc(x: int, y: int) -> str:
    return str(x + y)


async def frontend_show(message: str) -> str:
    return ""


def test_frontend_tool_calls_are_persisted_as_tool_messages() -> None:
    backend_tool_call_id = "call_backend_calc"
    frontend_tool_call_id = "call_frontend_show"

    def generate_response(messages: list[ChatMessage], **kwargs: object) -> ChatMessage:
        if any(message.role == MessageRole.TOOL for message in messages):
            return ChatMessage(role=MessageRole.ASSISTANT, content="done")

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            additional_kwargs={
                "tool_calls": [
                    ToolSelection(
                        tool_id=backend_tool_call_id,
                        tool_name="backend_calc",
                        tool_kwargs={"x": 2, "y": 3},
                    ),
                    ToolSelection(
                        tool_id=frontend_tool_call_id,
                        tool_name="frontend_show",
                        tool_kwargs={"message": "5"},
                    ),
                ]
            },
        )

    workflow = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(
            response_generator=generate_response,
            is_chat_model=True,
        ),
        backend_tools=[
            FunctionTool.from_defaults(
                async_fn=backend_calc,
                name="backend_calc",
                description="Add numbers server-side",
            )
        ],
        frontend_tools=[
            FunctionTool.from_defaults(
                async_fn=frontend_show,
                name="frontend_show",
                description="Show a message in the UI",
            )
        ],
        timeout=30,
    )

    async def run_workflow() -> list[ChatMessage]:
        handler = workflow.run(
            input_data=RunAgentInput(
                threadId="thread",
                runId="run",
                state={},
                tools=[],
                context=[],
                forwardedProps={},
                messages=[
                    UserMessage(
                        id="user_msg",
                        role="user",
                        content="Compute 2+3 and show the result.",
                    )
                ],
            )
        )
        async for _ in handler.stream_events():
            pass
        await handler
        return await handler.ctx.store.get("chat_history")

    chat_history = asyncio.run(run_workflow())
    claimed_tool_call_ids = {
        tool_call.tool_id
        for message in chat_history
        if message.role == MessageRole.ASSISTANT
        for tool_call in (message.additional_kwargs or {}).get("tool_calls", [])
    }
    replied_tool_call_ids = {
        (message.additional_kwargs or {}).get("tool_call_id")
        for message in chat_history
        if message.role == MessageRole.TOOL
    }

    assert claimed_tool_call_ids == {
        backend_tool_call_id,
        frontend_tool_call_id,
    }
    assert claimed_tool_call_ids <= replied_tool_call_ids
