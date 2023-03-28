"""LlamaIndex Tool classes."""

from typing import Dict, List, cast

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from gpt_index.indices.composability.graph import QUERY_CONFIG_TYPE, ComposableGraph
from gpt_index.indices.base import BaseGPTIndex


class IndexToolConfig(BaseModel):
    """Configuration for LlamaIndex index tool."""

    index: BaseGPTIndex
    name: str
    description: str
    index_query_kwargs: Dict = Field(default_factory=dict)
    tool_kwargs: Dict = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class GraphToolConfig(BaseModel):
    """Configuration for LlamaIndex graph tool."""

    graph: ComposableGraph
    name: str
    description: str
    query_configs: List[Dict] = Field(default_factory=dict)
    tool_kwargs: Dict = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class LlamaIndexTool(BaseTool):
    """Tool for querying a LlamaIndex."""

    # NOTE: name/description still needs to be set
    index: BaseGPTIndex
    query_kwargs: Dict = Field(default_factory=dict)

    @classmethod
    def from_tool_config(cls, tool_config: IndexToolConfig) -> "LlamaIndexTool":
        """Create a tool from a tool config."""
        return cls(
            index=tool_config.index,
            name=tool_config.name,
            description=tool_config.description,
            query_kwargs=tool_config.index_query_kwargs,
            **tool_config.tool_kwargs,
        )

    def _run(self, tool_input: str) -> str:
        response = self.index.query(tool_input, **self.query_kwargs)
        return str(response)

    async def _arun(self, tool_input: str) -> str:
        response = await self.index.aquery(tool_input, **self.query_kwargs)
        return str(response)


class LlamaGraphTool(BaseTool):
    """Tool for querying a ComposableGraph."""

    # NOTE: name/description still needs to be set
    graph: ComposableGraph
    query_configs: List[Dict] = Field(default_factory=list)

    @classmethod
    def from_tool_config(cls, tool_config: GraphToolConfig) -> "LlamaGraphTool":
        """Create a tool from a tool config."""
        return cls(
            graph=tool_config.graph,
            name=tool_config.name,
            description=tool_config.description,
            query_configs=tool_config.query_configs,
            **tool_config.tool_kwargs,
        )

    def _run(self, tool_input: str) -> str:
        query_configs = cast(List[QUERY_CONFIG_TYPE], self.query_configs)
        response = self.graph.query(tool_input, query_configs=query_configs)
        return str(response)

    async def _arun(self, tool_input: str) -> str:
        query_configs = cast(List[QUERY_CONFIG_TYPE], self.query_configs)
        response = await self.graph.aquery(tool_input, query_configs=query_configs)
        return str(response)
