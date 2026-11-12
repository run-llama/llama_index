"""
High-level A2A utility functions for LlamaIndex integration.

Provides convenience functions for:
- Discovering tools from an A2A agent card
"""

from typing import Any, Dict, List, Optional

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow
from pydantic import BaseModel

from llama_index.tools.a2a.base import A2AToolSpec
from llama_index.tools.a2a.client import BasicA2AClient


def get_tools_from_a2a_card(
    agent_card_url: str,
    allowed_tools: Optional[List[str]] = None,
    global_partial_params: Optional[Dict[str, Any]] = None,
    partial_params_by_tool: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[FunctionTool]:
    """
    Get tools from an A2A agent card (synchronous).

    Connects to the given AgentCard URL, discovers available skills,
    and returns them as LlamaIndex FunctionTools.

    Args:
        agent_card_url: URL of the agent card (e.g., https://agent.example.com/.well-known/agent-card.json).
        allowed_tools: If set, only return tools with the specified names.
        global_partial_params: A dict of params to apply to all tools globally.
        partial_params_by_tool: A dict mapping tool names to param overrides.

    Returns:
        A list of FunctionTool objects.

    Example:
        >>> tools = get_tools_from_a2a_card("https://weather-agent.example.com/.well-known/agent-card.json")
        >>> # Use tools in a LlamaIndex workflow
    """
    client = BasicA2AClient(agent_card_url)
    tool_spec = A2AToolSpec(
        client,
        allowed_tools=allowed_tools,
        global_partial_params=global_partial_params,
        partial_params_by_tool=partial_params_by_tool,
    )
    return tool_spec.to_tool_list()


async def aget_tools_from_a2a_card(
    agent_card_url: str,
    allowed_tools: Optional[List[str]] = None,
    global_partial_params: Optional[Dict[str, Any]] = None,
    partial_params_by_tool: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[FunctionTool]:
    """
    Get tools from an A2A agent card (asynchronous).

    This is the preferred method when working within async contexts.

    Args:
        agent_card_url: URL of the agent card.
        allowed_tools: If set, only return tools with the specified names.
        global_partial_params: A dict of params to apply to all tools globally.
        partial_params_by_tool: A dict mapping tool names to param overrides.

    Returns:
        A list of FunctionTool objects.

    Example:
        >>> tools = await aget_tools_from_a2a_card("https://search-agent.example.com/.well-known/agent-card.json")
        >>> print(f"Found {len(tools)} tools")
    """
    client = BasicA2AClient(agent_card_url)
    tool_spec = A2AToolSpec(
        client,
        allowed_tools=allowed_tools,
        global_partial_params=global_partial_params,
        partial_params_by_tool=partial_params_by_tool,
    )
    return await tool_spec.to_tool_list_async()
