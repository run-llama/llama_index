"""Tool mapping."""

from typing import Any, Optional, Sequence

from llama_index.objects.base_node_mapping import BaseObjectNodeMapping
from llama_index.schema import BaseNode, TextNode
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools.types import BaseTool


def convert_tool_to_node(tool: BaseTool) -> TextNode:
    """Function convert Tool to node."""
    node_text = (
        f"Tool name: {tool.metadata.name}\n"
        f"Tool description: {tool.metadata.description}\n"
    )
    if tool.metadata.fn_schema is not None:
        node_text += f"Tool schema: {tool.metadata.fn_schema.schema()}\n"
    return TextNode(
        text=node_text,
        metadata={"name": tool.metadata.name},
        excluded_embed_metadata_keys=["name"],
        excluded_llm_metadata_keys=["name"],
    )


class BaseToolNodeMapping(BaseObjectNodeMapping[BaseTool]):
    """Base Tool node mapping."""

    def validate_object(self, obj: BaseTool) -> None:
        if not isinstance(obj, BaseTool):
            raise ValueError(f"Object must be of type {BaseTool}")


class SimpleToolNodeMapping(BaseToolNodeMapping):
    """Simple Tool mapping.

    In this setup, we assume that the tool name is unique, and
    that the list of all tools are stored in memory.

    """

    def __init__(self, objs: Optional[Sequence[BaseTool]] = None) -> None:
        objs = objs or []
        self._tools = {tool.metadata.name: tool for tool in objs}

    @classmethod
    def from_objects(
        cls, objs: Sequence[BaseTool], *args: Any, **kwargs: Any
    ) -> "BaseObjectNodeMapping":
        return cls(objs)

    def _add_object(self, tool: BaseTool) -> None:
        self._tools[tool.metadata.name] = tool

    def to_node(self, tool: BaseTool) -> TextNode:
        """To node."""
        return convert_tool_to_node(tool)

    def _from_node(self, node: BaseNode) -> BaseTool:
        """From node."""
        if node.metadata is None:
            raise ValueError("Metadata must be set")
        return self._tools[node.metadata["name"]]


class BaseQueryToolNodeMapping(BaseObjectNodeMapping[QueryEngineTool]):
    """Base query tool node mapping."""


class SimpleQueryToolNodeMapping(BaseQueryToolNodeMapping):
    """Simple query tool mapping."""

    def __init__(self, objs: Optional[Sequence[QueryEngineTool]] = None) -> None:
        objs = objs or []
        self._tools = {tool.metadata.name: tool for tool in objs}

    def validate_object(self, obj: QueryEngineTool) -> None:
        if not isinstance(obj, QueryEngineTool):
            raise ValueError(f"Object must be of type {QueryEngineTool}")

    @classmethod
    def from_objects(
        cls, objs: Sequence[QueryEngineTool], *args: Any, **kwargs: Any
    ) -> "BaseObjectNodeMapping":
        return cls(objs)

    def _add_object(self, tool: QueryEngineTool) -> None:
        if tool.metadata.name is None:
            raise ValueError("Tool name must be set")
        self._tools[tool.metadata.name] = tool

    def to_node(self, obj: QueryEngineTool) -> TextNode:
        """To node."""
        return convert_tool_to_node(obj)

    def _from_node(self, node: BaseNode) -> QueryEngineTool:
        """From node."""
        if node.metadata is None:
            raise ValueError("Metadata must be set")
        return self._tools[node.metadata["name"]]
