"""
The server's ``system_prompt`` must not leak into ``MESSAGES_SNAPSHOT``.

AG-UI clients adopt the snapshot as the thread and send it back on the next
run. If the injected prompt is part of the snapshot, the next run injects it
again: appended to the echoed system message (one more copy per run) or, when
the echo isn't first, as a second system message.
"""

import asyncio
from typing import List, Tuple

from ag_ui.core import Message, RunAgentInput, SystemMessage, UserMessage

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow
from llama_index.protocols.ag_ui.events import MessagesSnapshotWorkflowEvent

PROMPT = "You are a helpful assistant."


def run(messages: List[Message]) -> Tuple[List[ChatMessage], List[Message]]:
    """One run; returns what the LLM received and the snapshot sent to the client."""
    seen: List[ChatMessage] = []

    def respond(chat: List[ChatMessage], **kwargs: object) -> ChatMessage:
        seen[:] = chat
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")

    workflow = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(response_generator=respond, is_chat_model=True),
        system_prompt=PROMPT,
        timeout=30,
    )

    async def go() -> List[Message]:
        handler = workflow.run(
            input_data=RunAgentInput(
                threadId="thread",
                runId="run",
                state={},
                tools=[],
                context=[],
                forwardedProps={},
                messages=messages,
            )
        )
        snapshot: List[Message] = []
        async for ev in handler.stream_events():
            if isinstance(ev, MessagesSnapshotWorkflowEvent):
                snapshot = ev.messages
        await handler
        return snapshot

    snapshot = asyncio.run(go())
    return list(seen), snapshot


def system_texts(messages: List[ChatMessage]) -> List[str]:
    return [m.content or "" for m in messages if m.role == MessageRole.SYSTEM]


def test_system_prompt_is_sent_once_per_run_and_not_snapshotted() -> None:
    thread: List[Message] = []
    for i in range(3):
        thread = [
            *thread,
            UserMessage(id=f"u{i}", role="user", content=f"message {i}"),
        ]
        seen, thread = run(thread)
        assert system_texts(seen) == [PROMPT]
        assert seen[0].role == MessageRole.SYSTEM
        assert [m.role for m in thread] == ["user", "assistant"] * (i + 1)


def test_client_system_message_is_kept_without_the_server_prompt() -> None:
    thread: List[Message] = [
        SystemMessage(id="client_sys", role="system", content="Client rules."),
    ]
    for i in range(3):
        thread = [
            *thread,
            UserMessage(id=f"u{i}", role="user", content=f"message {i}"),
        ]
        seen, thread = run(thread)
        assert system_texts(seen) == [f"Client rules.\n{PROMPT}"]
        system = [m for m in thread if m.role == "system"]
        assert [(m.id, m.content) for m in system] == [("client_sys", "Client rules.")]
