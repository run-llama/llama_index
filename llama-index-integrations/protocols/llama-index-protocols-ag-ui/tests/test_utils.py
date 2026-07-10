import pytest

from llama_index.core.llms import ChatMessage
from llama_index.protocols.ag_ui.utils import (
    ag_ui_message_to_llama_index_message,
    llama_index_message_to_ag_ui_message,
)


def test_tool_message_missing_tool_call_id_raises():
    """
    Regression test for GH #22068: a 'tool' role ChatMessage without
    'tool_call_id' in additional_kwargs used to silently fabricate a random
    uuid4 as ToolMessage.tool_call_id. That id pairs with no
    AssistantMessage.tool_calls[*].id, which providers like OpenAI/Anthropic
    reject as an invalid message history. It must now raise instead of
    silently producing a broken message.
    """
    message = ChatMessage(
        role="tool",
        content="22 C, sunny",
        additional_kwargs={"id": "tool_msg_1"},  # no tool_call_id
    )
    with pytest.raises(ValueError, match="tool_call_id"):
        llama_index_message_to_ag_ui_message(message)


def test_tool_message_with_tool_call_id_round_trips():
    """A tool message that does carry tool_call_id must convert cleanly and
    round-trip back with the same id, matching the AssistantMessage that
    issued the call."""
    message = ChatMessage(
        role="tool",
        content="22 C, sunny",
        additional_kwargs={"id": "tool_msg_1", "tool_call_id": "call_get_weather_001"},
    )

    ag_ui_message = llama_index_message_to_ag_ui_message(message)
    assert ag_ui_message.tool_call_id == "call_get_weather_001"

    back = ag_ui_message_to_llama_index_message(ag_ui_message)
    assert back.additional_kwargs.get("tool_call_id") == "call_get_weather_001"


def test_full_history_missing_tool_call_id_raises_not_silently_orphaned():
    """End-to-end shape from the bug report: an assistant message with a real
    tool call id, followed by a tool message that lost its tool_call_id
    (e.g. dropped by a memory backend on serialize/restore). Converting the
    full history must fail loudly instead of producing a ToolMessage whose
    tool_call_id doesn't match the AssistantMessage's tool call id."""
    history = [
        ChatMessage(
            role="assistant",
            content="",
            additional_kwargs={
                "id": "asst_msg_1",
                "ag_ui_tool_calls": [
                    {"id": "call_get_weather_001", "name": "get_weather", "arguments": '{"city": "Paris"}'}
                ],
            },
        ),
        ChatMessage(
            role="tool",
            content="22 C, sunny",
            additional_kwargs={"id": "tool_msg_1"},  # tool_call_id dropped
        ),
    ]

    llama_index_message_to_ag_ui_message(history[0])  # assistant message converts fine
    with pytest.raises(ValueError, match="tool_call_id"):
        llama_index_message_to_ag_ui_message(history[1])
