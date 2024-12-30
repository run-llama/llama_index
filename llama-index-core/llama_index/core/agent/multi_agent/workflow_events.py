from typing import Any, Optional

from llama_index.core.tools import AsyncBaseTool, ToolSelection, ToolOutput
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event
from llama_index.core.agent.multi_agent.agent_config import AgentConfig


class ToolApprovalNeeded(Event):
    """Emitted when a tool call needs approval."""

    id: str
    tool_name: str
    tool_kwargs: dict


class ApproveTool(Event):
    """Required to approve a tool."""

    id: str
    tool_name: str
    tool_kwargs: dict
    approved: bool
    reason: Optional[str] = None


class AgentInput(Event):
    """LLM input."""

    input: list[ChatMessage]
    current_agent: str


class AgentSetup(Event):
    """Agent setup."""

    input: list[ChatMessage]
    current_agent: str
    current_config: AgentConfig
    tools: list[AsyncBaseTool]


class AgentStream(Event):
    """Agent stream."""

    delta: str
    current_agent: str
    tool_calls: list[ToolSelection]
    raw_response: Any


class AgentOutput(Event):
    """LLM output."""

    response: str
    tool_calls: list[ToolSelection]
    raw_response: Any
    current_agent: str


class ToolCall(Event):
    """All tool calls are surfaced."""

    tool_name: str
    tool_kwargs: dict
    tool_id: str


class ToolCallResult(ToolCall):
    """Tool call result."""

    tool_output: ToolOutput
    return_direct: bool
