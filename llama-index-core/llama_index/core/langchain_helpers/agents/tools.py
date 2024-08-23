"""LlamaIndex Tool classes."""

from typing import Any, Dict, List

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.bridge.langchain import BaseTool
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.schema import TextNode


def _get_response_with_sources(response: RESPONSE_TYPE) -> str:
    """Return a response with source node info."""
    source_data: List[Dict[str, Any]] = []
    for source_node in response.source_nodes:
        metadata = {}
        if isinstance(source_node.node, TextNode):
            start = source_node.node.start_char_idx
            end = source_node.node.end_char_idx
            if start is not None and end is not None:
                metadata.update({"start_char_idx": start, "end_char_idx": end})

        source_data.append(metadata)
        source_data[-1]["ref_doc_id"] = source_node.node.ref_doc_id
        source_data[-1]["score"] = source_node.score
    return str({"answer": str(response), "sources": source_data})


class IndexToolConfig(BaseModel):
    """Configuration for LlamaIndex index tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    query_engine: BaseQueryEngine
    name: str
    description: str
    tool_kwargs: Dict = Field(default_factory=dict)


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

    def _run(self, input: str) -> str:
        response = self.query_engine.query(input)
        if self.return_sources:
            return _get_response_with_sources(response)
        return str(response)

    async def _arun(self, input: str) -> str:
        response = await self.query_engine.aquery(input)
        if self.return_sources:
            return _get_response_with_sources(response)
        return str(response)
