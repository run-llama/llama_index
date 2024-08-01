"""Utils for OpenAI agent."""

from typing import List, Union

from llama_index.core.tools import BaseTool


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


def resolve_tool_choice(tool_choice: Union[str, dict] = "auto") -> Union[str, dict]:
    """Resolve tool choice.

    If tool_choice is a function name string, return the appropriate dict.
    """
    if isinstance(tool_choice, str) and tool_choice not in ["none", "auto"]:
        return {"type": "function", "function": {"name": tool_choice}}

    return tool_choice
