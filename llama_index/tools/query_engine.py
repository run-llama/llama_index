from typing import TYPE_CHECKING, Any, Optional

from llama_index.core import BaseQueryEngine

if TYPE_CHECKING:
    from llama_index.langchain_helpers.agents.tools import (
        LlamaIndexTool,
    )
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput

DEFAULT_NAME = "query_engine_tool"
DEFAULT_DESCRIPTION = """Useful for running a natural language query
against a knowledge base and get back a natural language response.
"""


class QueryEngineTool(AsyncBaseTool):
    """Query engine tool.

    A tool making use of a query engine.

    Args:
        query_engine (BaseQueryEngine): A query engine.
        metadata (ToolMetadata): The associated metadata of the query engine.
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        metadata: ToolMetadata,
        resolve_input_errors: bool = True,
    ) -> None:
        self._query_engine = query_engine
        self._metadata = metadata
        self._resolve_input_errors = resolve_input_errors

    @classmethod
    def from_defaults(
        cls,
        query_engine: BaseQueryEngine,
        name: Optional[str] = None,
        description: Optional[str] = None,
        resolve_input_errors: bool = True,
    ) -> "QueryEngineTool":
        name = name or DEFAULT_NAME
        description = description or DEFAULT_DESCRIPTION

        metadata = ToolMetadata(name=name, description=description)
        return cls(
            query_engine=query_engine,
            metadata=metadata,
            resolve_input_errors=resolve_input_errors,
        )

    @property
    def query_engine(self) -> BaseQueryEngine:
        return self._query_engine

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        if args is not None and len(args) > 0:
            query_str = str(args[0])
        elif kwargs is not None and "input" in kwargs:
            # NOTE: this assumes our default function schema of `input`
            query_str = kwargs["input"]
        elif kwargs is not None and self._resolve_input_errors:
            query_str = str(kwargs)
        else:
            raise ValueError(
                "Cannot call query engine without specifying `input` parameter."
            )

        response = self._query_engine.query(query_str)
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"input": input},
            raw_output=response,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        if args is not None and len(args) > 0:
            query_str = str(args[0])
        elif kwargs is not None and "input" in kwargs:
            # NOTE: this assumes our default function schema of `input`
            query_str = kwargs["input"]
        elif kwargs is not None and self._resolve_input_errors:
            query_str = str(kwargs)
        else:
            raise ValueError("Cannot call query engine without inputs")

        response = await self._query_engine.aquery(query_str)
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"input": input},
            raw_output=response,
        )

    def as_langchain_tool(self) -> "LlamaIndexTool":
        from llama_index.langchain_helpers.agents.tools import (
            IndexToolConfig,
            LlamaIndexTool,
        )

        tool_config = IndexToolConfig(
            query_engine=self.query_engine,
            name=self.metadata.name,
            description=self.metadata.description,
        )
        return LlamaIndexTool.from_tool_config(tool_config=tool_config)
