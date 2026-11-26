"""
``RunAgentInput.context`` must reach the LLM.

AG-UI clients send app context there (CopilotKit fills it from
``useAgentContext``). It is rendered into the system message the LLM receives
on every LLM call of the run, and kept out of ``MESSAGES_SNAPSHOT`` so the
client never adopts it as part of the thread.
"""

import asyncio
from typing import Any, List, Optional, Tuple

from ag_ui.core import Context, Message, RunAgentInput, UserMessage

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow
from llama_index.protocols.ag_ui.events import MessagesSnapshotWorkflowEvent

CONTEXT = [
    Context(description="Current page", value="Order #1234"),
    Context(description="User", value="Signed in as Ada"),
]
RENDERED = "## Current page\nOrder #1234\n\n## User\nSigned in as Ada"
USER = UserMessage(id="u1", role="user", content="What am I looking at?")


async def lookup(order: str) -> str:
    return "shipped"


def run(
    context: List[Context],
    system_prompt: Optional[str] = None,
    call_backend_tool: bool = False,
) -> Tuple[List[List[ChatMessage]], List[Message]]:
    """One run; returns the messages of every LLM call and the final snapshot."""
    calls: List[List[ChatMessage]] = []

    def respond(chat: List[ChatMessage], **kwargs: Any) -> ChatMessage:
        calls.append(list(chat))
        if call_backend_tool and len(calls) == 1:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="call_1",
                            tool_name="lookup",
                            tool_kwargs={"order": "1234"},
                        )
                    ]
                },
            )
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")

    workflow = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(response_generator=respond, is_chat_model=True),
        backend_tools=[FunctionTool.from_defaults(async_fn=lookup, name="lookup")],
        system_prompt=system_prompt,
        timeout=30,
    )

    async def go() -> List[Message]:
        handler = workflow.run(
            input_data=RunAgentInput(
                threadId="thread",
                runId="run",
                state={},
                tools=[],
                context=context,
                forwardedProps={},
                messages=[USER],
            )
        )
        snapshot: List[Message] = []
        async for ev in handler.stream_events():
            if isinstance(ev, MessagesSnapshotWorkflowEvent):
                snapshot = ev.messages
        await handler
        return snapshot

    snapshot = asyncio.run(go())
    return calls, snapshot


def system_texts(messages: List[ChatMessage]) -> List[str]:
    return [m.content or "" for m in messages if m.role == MessageRole.SYSTEM]


def test_context_is_appended_to_the_system_prompt() -> None:
    (call,), _ = run(CONTEXT, system_prompt="You are helpful.")
    assert system_texts(call) == [f"You are helpful.\n{RENDERED}"]
    assert call[0].role == MessageRole.SYSTEM


def test_context_without_a_system_prompt_gets_its_own_system_message() -> None:
    (call,), _ = run(CONTEXT)
    assert system_texts(call) == [RENDERED]
    assert call[0].role == MessageRole.SYSTEM
    assert call[1].content == "What am I looking at?"


def test_context_reaches_every_llm_call_of_the_run() -> None:
    calls, _ = run(CONTEXT, call_backend_tool=True)
    assert len(calls) == 2
    for call in calls:
        assert system_texts(call) == [RENDERED]


def test_context_is_not_part_of_the_snapshot() -> None:
    _, snapshot = run(CONTEXT, call_backend_tool=True)
    assert snapshot
    for message in snapshot:
        assert "Order #1234" not in str(getattr(message, "content", ""))


def test_no_context_leaves_the_messages_unchanged() -> None:
    (call,), _ = run([])
    assert system_texts(call) == []
    assert [m.content for m in call] == ["What am I looking at?"]
