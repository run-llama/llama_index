"""Tool mapping"""

from llama_index.objects.base_node_mapping import BaseObjectNodeMapping
from llama_index.tools.types import BaseTool
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.data_structs.node import Node
from typing import Sequence, TypeVar, Any


def convert_tool_to_node(tool: BaseTool) -> Node:
    """Function convert Tool to node."""
    node_text = (
        f"Tool name: {tool.metadata.name}\n"
        f"Tool description: {tool.metadata.description}\n"
    )
    if tool.metadata.fn_schema is not None:
        node_text += f"Tool schema: {tool.metadata.fn_schema.schema()}\n"
    return Node(text=node_text, node_info={"name": tool.metadata.name})


class BaseToolNodeMapping(BaseObjectNodeMapping[BaseTool]):
    """Base Tool node mapping."""


class SimpleToolNodeMapping(BaseToolNodeMapping):
    """Simple Tool mapping.

    In this setup, we assume that the tool name is unique, and
    that the list of all tools are stored in memory.

    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self._tools = {tool.metadata.name: tool for tool in tools}

    @classmethod
    def from_objects(
        cls, tools: Sequence[BaseTool], *args: Any, **kwargs: Any
    ) -> "SimpleToolNodeMapping":
        return cls(tools)

    def add_object(self, tool: BaseTool) -> None:
        self._tools[tool.metadata.name] = tool

    def to_node(self, tool: BaseTool) -> BaseTool:
        """To node."""
        return convert_tool_to_node(tool)

    def from_node(self, node: Node) -> BaseTool:
        """From node."""
        return self._tools[node.node_info["name"]]


class BaseQueryToolNodeMapping(BaseObjectNodeMapping[QueryEngineTool]):
    """Base query tool node mapping."""


class SimpleQueryToolNodeMapping(BaseQueryToolNodeMapping):
    """Simple query tool mapping."""

    def __init__(self, tools: Sequence[QueryEngineTool]) -> None:
        self._tools = {hash(tool): tool for tool in tools}

    @classmethod
    def from_objects(
        cls, tools: Sequence[QueryEngineTool]
    ) -> "SimpleQueryToolNodeMapping":
        return cls(tools)

    def add_object(self, tool: QueryEngineTool) -> None:
        self._objs[tool.metadata.name] = tool

    def to_node(self, obj: QueryEngineTool) -> QueryEngineTool:
        """To node."""
        return convert_tool_to_node(obj)

    def from_node(self, node: Node) -> QueryEngineTool:
        """From node."""
        return self._tools[node.node_info["name"]]
