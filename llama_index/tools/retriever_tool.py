"""Retriever tool."""

from typing import Any, Optional, cast

from llama_index.indices.base_retriever import BaseRetriever
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.langchain_helpers.agents.tools import LlamaIndexTool

DEFAULT_NAME = "retriever_tool"
DEFAULT_DESCRIPTION = """Useful for running a natural language query
against a knowledge base and retrieving a set of relevant documents.
"""


class RetrieverTool(AsyncBaseTool):
    """Retriever tool.

    A tool making use of a retriever.

    Args:
        retriever (BaseRetriever): A retriever.
        metadata (ToolMetadata): The associated metadata of the query engine.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        metadata: ToolMetadata,
    ) -> None:
        self._retriever = retriever
        self._metadata = metadata

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "RetrieverTool":
        name = name or DEFAULT_NAME
        description = description or DEFAULT_DESCRIPTION

        metadata = ToolMetadata(name=name, description=description)
        return cls(retriever=retriever, metadata=metadata)

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, tool_input: Any) -> ToolOutput:
        query_str = cast(str, tool_input)
        docs = self._retriever.retrieve(query_str)
        return ToolOutput(
            content=str(docs),
            tool_name=self.metadata.name,
            raw_input={"input": tool_input},
            raw_output=docs,
        )

    async def acall(self, tool_input: Any) -> ToolOutput:
        query_str = cast(str, tool_input)
        docs = await self._retriever.aretrieve(query_str)
        return ToolOutput(
            content=str(docs),
            tool_name=self.metadata.name,
            raw_input={"input": tool_input},
            raw_output=docs,
        )

    def as_langchain_tool(self) -> LlamaIndexTool:
        raise NotImplementedError("`as_langchain_tool` not implemented here.")
