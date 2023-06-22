"""LlamaIndex Tool classes."""

from typing import Dict

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from llama_index.indices.query.base import BaseQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.schema import TextNode


def _get_response_with_sources(response: RESPONSE_TYPE) -> str:
    """Return a response with source node info."""
    source_data = []
    for source_node in response.source_nodes:
        metadata = {}
        if isinstance(source_node.node, TextNode):
            node_info = source_node.node.node_info
            if node_info["start"] is not None and node_info["end"] is not None:
                metadata.update(node_info)

        source_data.append(metadata)
        source_data[-1]["ref_doc_id"] = source_node.node.ref_doc_id
        source_data[-1]["score"] = source_node.score
    return str({"answer": str(response), "sources": source_data})


class IndexToolConfig(BaseModel):
    """Configuration for LlamaIndex index tool."""

    query_engine: BaseQueryEngine
    name: str
    description: str
    tool_kwargs: Dict = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class LlamaIndexTool(BaseTool):
    """Tool for querying a LlamaIndex."""

    # NOTE: name/description still needs to be set
    query_engine: BaseQueryEngine
    return_sources: bool = False

    @classmethod
    def from_tool_config(cls, tool_config: IndexToolConfig) -> "LlamaIndexTool":
        """Create a tool from a tool config."""
        return_sources = tool_config.tool_kwargs.pop("return_sources", False)
        return cls(
            query_engine=tool_config.query_engine,
            name=tool_config.name,
            description=tool_config.description,
            return_sources=return_sources,
            **tool_config.tool_kwargs,
        )

    def _run(self, tool_input: str) -> str:
        response = self.query_engine.query(tool_input)
        if self.return_sources:
            return _get_response_with_sources(response)
        return str(response)

    async def _arun(self, tool_input: str) -> str:
        response = await self.query_engine.aquery(tool_input)
        if self.return_sources:
            return _get_response_with_sources(response)
        return str(response)
