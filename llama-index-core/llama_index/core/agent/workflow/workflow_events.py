from typing import Any

from llama_index.core.bridge.pydantic import model_serializer
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event, StartEvent


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
    raw: Any


class AgentOutput(Event):
    """LLM output."""

    response: ChatMessage
    tool_calls: list[ToolSelection]
    raw: Any
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
    @model_serializer()
    def serialize_start_event(self) -> dict:
        """Serialize the start event and exclude the memory."""
        return {
            "user_msg": self.user_msg,
            "chat_history": self.chat_history,
        }
