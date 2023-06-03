"""Index tool.

This tool is used to call an existing tool, but then index the output using
LlamaIndex data structures, and then query it.

"""

from abc import abstractmethod
from llama_index.readers.schema.base import Document
from llama_index.tools.types import BaseTool, ToolMetadata
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from typing import Any, Type, Optional


class IndexTool(BaseTool):
    """Index tool."""

    def __init__(
        self,
        tool: BaseTool,
        tool_metadata: ToolMetadata,
        index_cls: Type[BaseGPTIndex],
        **index_kwargs: Any,
    ) -> None:
        """Initialize with tool."""
        self._tool = tool
        self._index_cls = index_cls
        self._index_kwargs = index_kwargs
        self._metadata = tool_metadata

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @classmethod
    def from_defaults(
        cls,
        tool: BaseTool,
        tool_metadata: Optional[ToolMetadata] = None,
        index_cls: Optional[Type[BaseGPTIndex]] = None,
        **index_kwargs: Any,
    ) -> "IndexTool":
        """Initialize from defaults."""
        index_cls = index_cls or GPTVectorStoreIndex
        tool_metadata = tool_metadata or tool.metadata
        return cls(
            tool=tool, tool_metadata=tool_metadata, index_cls=index_cls, **index_kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Call."""
        output = self._tool(*args, **kwargs)
        index = self._index_cls.from_documents([Document(output)], **self._index_kwargs)
        query_engine = index.as_query_engine()

        return query_engine.query(output)
