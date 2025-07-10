import logging
from typing import Any, Optional

from llama_index.core.bridge.pydantic import Field, model_serializer, ValidationError
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event, StartEvent


logger = logging.getLogger(__name__)


class AgentInput(Event):
    """LLM input."""

    input: list[ChatMessage]
    current_agent_name: str


class AgentSetup(Event):
    """Agent setup."""

    input: list[ChatMessage]
    current_agent_name: str


class AgentStream(Event):
    """Agent stream."""

    delta: str
    response: str
    current_agent_name: str
    tool_calls: list[ToolSelection]
    raw: Optional[Any] = Field(default=None, exclude=True)


class AgentOutput(Event):
    """LLM output."""

    response: ChatMessage
    tool_calls: list[ToolSelection]
    raw: Optional[Any] = Field(default=None, exclude=True)
    current_agent_name: str

    def __str__(self) -> str:
        return self.response.content or ""


class ToolCall(Event):
    """All tool calls are surfaced."""

    tool_name: str
    tool_kwargs: dict
    tool_id: str


class ToolCallResult(Event):
    """Tool call result."""

    tool_name: str
    tool_kwargs: dict
    tool_id: str
    tool_output: ToolOutput
    return_direct: bool


class AgentWorkflowStartEvent(StartEvent):
    def __init__(self, **data: Any) -> None:
        """Convert chat_history items to ChatMessage objects if they aren't already"""
        if "chat_history" in data and data["chat_history"]:
            converted_history = []
            for i, msg in enumerate(data["chat_history"]):
                if isinstance(msg, ChatMessage):
                    converted_history.append(msg)
                else:
                    # Convert dict or other formats to ChatMessage with validation
                    try:
                        converted_history.append(ChatMessage.model_validate(msg))
                    except ValidationError as e:
                        logger.error(
                            f"Failed to validate chat message at index {i}: {e}. "
                            f"Invalid message: {msg}"
                        )
                        raise
            data["chat_history"] = converted_history

        super().__init__(**data)

    @model_serializer()
    def serialize_start_event(self) -> dict:
        """Serialize the start event and exclude the memory."""
        return {
            "user_msg": self.user_msg,
            "chat_history": self.chat_history,
            "max_iterations": self.max_iterations,
        }
