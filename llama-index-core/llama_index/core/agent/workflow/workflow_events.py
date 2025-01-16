from typing import Any

from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event


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


class ToolCallResult(ToolCall):
    """Tool call result."""

    tool_output: ToolOutput
    return_direct: bool
