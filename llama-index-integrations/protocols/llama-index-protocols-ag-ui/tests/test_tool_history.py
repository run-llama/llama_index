"""
Earlier tool calls and results sent by the client must reach the LLM as native
tool calls (``ToolCallBlock``) and ``tool`` messages, not as text.

Models imitate the history they're given: with tool calls rendered as
``<tool_call>`` text, they answer later turns with tool-call-shaped text
instead of a tool call.
"""

import asyncio
import inspect
from typing import Any, List, Tuple

import pytest

from ag_ui.core import (
    AssistantMessage,
    FunctionCall,
    Message,
    RunAgentInput,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow
from llama_index.protocols.ag_ui.events import MessagesSnapshotWorkflowEvent
from llama_index.protocols.ag_ui.utils import (
    ag_ui_message_to_llama_index_message,
    llama_index_message_to_ag_ui_message,
)

CALL = AssistantMessage(
    id="a1",
    role="assistant",
    content="Checking.",
    tool_calls=[
        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"city":"Paris"}'),
        ),
        ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="get_time", arguments="{}"),
        ),
    ],
)
RESULT = ToolMessage(
    id="t1", role="tool", tool_call_id="call_1", content='{"temp_c":18}'
)


def test_assistant_tool_calls_become_tool_call_blocks() -> None:
    message = ag_ui_message_to_llama_index_message(CALL)

    assert message.role == MessageRole.ASSISTANT
    assert isinstance(message.blocks[0], TextBlock)
    assert message.blocks[0].text == "Checking."
    calls = [b for b in message.blocks if isinstance(b, ToolCallBlock)]
    assert [(c.tool_call_id, c.tool_name, c.tool_kwargs) for c in calls] == [
        ("call_1", "get_weather", {"city": "Paris"}),
        ("call_2", "get_time", {}),
    ]
    assert "<tool_call>" not in (message.content or "")


def test_tool_results_become_tool_messages() -> None:
    message = ag_ui_message_to_llama_index_message(RESULT)

    assert message.role == MessageRole.TOOL
    assert message.content == '{"temp_c":18}'
    assert message.additional_kwargs["tool_call_id"] == "call_1"


def test_tool_result_content_parts_become_text_in_a_tool_message() -> None:
    message = ag_ui_message_to_llama_index_message(
        ToolMessage(
            id="t2",
            role="tool",
            tool_call_id="call_1",
            content=[
                {"type": "text", "text": '{"temp_c":'},
                {"type": "text", "text": "18}"},
            ],
        )
    )
    assert message.role == MessageRole.TOOL
    assert message.content == '{"temp_c":18}'
    assert message.additional_kwargs["tool_call_id"] == "call_1"


def test_unparseable_arguments_are_kept_as_sent() -> None:
    call = AssistantMessage(
        id="a2",
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_3",
                type="function",
                function=FunctionCall(name="search", arguments='{"q": "unterminated'),
            )
        ],
    )
    (block,) = ag_ui_message_to_llama_index_message(call).blocks
    assert isinstance(block, ToolCallBlock)
    assert block.tool_kwargs == '{"q": "unterminated'


def test_conversion_round_trips() -> None:
    for original in (CALL, RESULT):
        converted = ag_ui_message_to_llama_index_message(original)
        assert llama_index_message_to_ag_ui_message(converted) == original


def run(messages: List[Message]) -> Tuple[List[ChatMessage], List[Message]]:
    # MockFunctionCallingLLM(response_generator=...) needs llama-index-core
    # 0.14.12; the converter tests above run down to the package's floor (0.14.5).
    mock = pytest.importorskip("llama_index.core.llms.mock")
    MockFunctionCallingLLM = getattr(mock, "MockFunctionCallingLLM", None)
    if MockFunctionCallingLLM is None or (
        "response_generator"
        not in inspect.signature(MockFunctionCallingLLM.__init__).parameters
    ):
        pytest.skip("MockFunctionCallingLLM(response_generator=...) needs core 0.14.12")

    seen: List[ChatMessage] = []

    def respond(chat: List[ChatMessage], **kwargs: Any) -> ChatMessage:
        seen[:] = chat
        return ChatMessage(role=MessageRole.ASSISTANT, content="It's 18°C.")

    workflow = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(response_generator=respond, is_chat_model=True),
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


def test_workflow_sends_native_tool_history_and_snapshots_it_unchanged() -> None:
    user = UserMessage(id="u1", role="user", content="Weather in Paris?")
    seen, snapshot = run([user, CALL, RESULT])

    assert [m.role for m in seen] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert [b.tool_call_id for b in seen[1].blocks if isinstance(b, ToolCallBlock)] == [
        "call_1",
        "call_2",
    ]
    assert seen[2].additional_kwargs["tool_call_id"] == "call_1"
    # The client gets its own messages back exactly: ids and argument strings.
    assert snapshot[:3] == [user, CALL, RESULT]
