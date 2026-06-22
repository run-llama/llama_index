import pytest
from ag_ui.core import ToolMessage

from llama_index.core.llms import ChatMessage
from llama_index.protocols.ag_ui.utils import llama_index_message_to_ag_ui_message


def test_tool_message_requires_tool_call_id() -> None:
    message = ChatMessage(
        role="tool",
        content="22 C, sunny",
        additional_kwargs={"id": "tool_msg_1"},
    )

    with pytest.raises(ValueError, match="tool_call_id"):
        llama_index_message_to_ag_ui_message(message)


def test_tool_message_preserves_tool_call_id() -> None:
    message = ChatMessage(
        role="tool",
        content="22 C, sunny",
        additional_kwargs={
            "id": "tool_msg_1",
            "tool_call_id": "call_get_weather_001",
        },
    )

    ag_ui_message = llama_index_message_to_ag_ui_message(message)

    assert isinstance(ag_ui_message, ToolMessage)
    assert ag_ui_message.tool_call_id == "call_get_weather_001"
