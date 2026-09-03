"""Reproduction + regression test for issue #22066.

Bug: AGUIChatWorkflow.aggregate_tool_calls only appended ToolMessages for
backend tool calls, orphaning frontend tool_call_ids. This broke the
LLM-provider pairing invariant when backend + frontend tools fired in the
same assistant turn.
"""
import asyncio
import sys
from uuid import uuid4

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


def make_workflow():
    backend = FunctionTool.from_defaults(
        async_fn=backend_calc,
        name="backend_calc",
        description="add x+y server-side",
    )
    frontend = FunctionTool.from_defaults(
        async_fn=frontend_show,
        name="frontend_show",
        description="render in UI (frontend)",
    )

    B = f"call_backend_{uuid4().hex[:8]}"
    F = f"call_frontend_{uuid4().hex[:8]}"

    def gen(messages, **kw):
        if any(m.role == MessageRole.TOOL for m in messages):
            return ChatMessage(role="assistant", content="(done)")
        return ChatMessage(
            role="assistant",
            content="",
            additional_kwargs={
                "tool_calls": [
                    ToolSelection(
                        tool_id=B,
                        tool_name="backend_calc",
                        tool_kwargs={"x": 2, "y": 3},
                    ),
                    ToolSelection(
                        tool_id=F,
                        tool_name="frontend_show",
                        tool_kwargs={"message": "5"},
                    ),
                ]
            },
        )

    wf = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(response_generator=gen, is_chat_model=True),
        backend_tools=[backend],
        frontend_tools=[frontend],
        timeout=30,
    )
    return wf, B, F


async def main() -> int:
    wf, B, F = make_workflow()

    handler = wf.run(
        input_data=RunAgentInput(
            threadId="t",
            runId="r",
            state={},
            tools=[],
            context=[],
            forwardedProps={},
            messages=[
                UserMessage(id="u1", role="user", content="Compute 2+3 and show.")
            ],
        )
    )
    async for _ in handler.stream_events():
        pass
    await handler

    history = await handler.ctx.store.get("chat_history")
    claimed = {
        tc.tool_id
        for m in history
        if m.role == MessageRole.ASSISTANT
        for tc in (m.additional_kwargs or {}).get("tool_calls", [])
    }
    replied = {
        (m.additional_kwargs or {}).get("tool_call_id")
        for m in history
        if m.role == MessageRole.TOOL
    }

    print(f"Claimed tool_call_ids: {claimed}")
    print(f"Replied tool_call_ids: {replied}")
    missing = claimed - replied
    if missing:
        print(f"FAIL: orphan tool_call_ids: {missing}")
        return 1
    print("PASS: every tool_call_id has a matching ToolMessage")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))