import re

from ag_ui.core import ToolMessage
from llama_index.core.llms import ChatMessage
from llama_index.protocols.ag_ui.utils import (
    llama_index_message_to_ag_ui_message,
)

UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def test_tool_message_uses_real_tool_call_id_from_additional_kwargs():
    """A tool ChatMessage carrying the real id must reuse it, not mint a uuid."""
    message = ChatMessage(
        role="tool",
        content="22 C, sunny",
        additional_kwargs={"id": "tool_msg_1", "tool_call_id": "call_get_weather_001"},
    )

    ag_ui_message = llama_index_message_to_ag_ui_message(message)

    assert isinstance(ag_ui_message, ToolMessage)
    assert ag_ui_message.tool_call_id == "call_get_weather_001"


def test_user_role_message_with_tool_call_id_uses_real_id():
    """A role="user" message that carries tool_call_id routes to ToolMessage and keeps the id."""
    message = ChatMessage(
        role="user",
        content="22 C, sunny",
        additional_kwargs={"id": "tool_msg_1", "tool_call_id": "call_get_weather_001"},
    )

    ag_ui_message = llama_index_message_to_ag_ui_message(message)

    assert isinstance(ag_ui_message, ToolMessage)
    assert ag_ui_message.tool_call_id == "call_get_weather_001"


def test_tool_message_without_tool_call_id_does_not_fabricate_random_uuid():
    """
    When the id is genuinely absent the fallback must be deterministic.

    The old behaviour minted a fresh ``uuid.uuid4()`` each call, producing a
    different orphan id every time and silently breaking the
    AssistantMessage.tool_calls[*].id <-> ToolMessage.tool_call_id pairing.
    The fallback must instead be stable and derived from the message itself.
    """
    message = ChatMessage(
        role="tool",
        content="22 C, sunny",
        additional_kwargs={"id": "tool_msg_1"},  # no tool_call_id
    )

    first = llama_index_message_to_ag_ui_message(message)
    second = llama_index_message_to_ag_ui_message(message)

    assert isinstance(first, ToolMessage)
    # Deterministic: converting the same message twice yields the same id.
    assert first.tool_call_id == second.tool_call_id
    # The fallback is the message's own id, not a freshly fabricated uuid4.
    assert first.tool_call_id == "tool_msg_1"
    assert not UUID4_RE.match(first.tool_call_id or "")
