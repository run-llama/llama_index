from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from llama_index.bridge.langchain import StructuredTool, Tool
from llama_index.bridge.pydantic import BaseModel


class DefaultToolFnSchema(BaseModel):
    """Default tool function Schema."""

    input: str


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema

    @property
    def fn_schema_str(self) -> str:
        """Get fn schema as string."""
        if self.fn_schema is None:
            raise ValueError("fn_schema is None.")
        return str(self.fn_schema.schema())

    def get_name(self) -> str:
        """Get name."""
        if self.name is None:
            raise ValueError("name is None.")
        return self.name

    def to_openai_function(self) -> Dict[str, Any]:
        """To OpenAI function."""
        if self.fn_schema is None:
            parameters = {
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
                "type": "object",
            }
        else:
            parameters = self.fn_schema.schema()

        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }


class ToolOutput(BaseModel):
    """Tool output."""

    content: str
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Any

    def __str__(self) -> str:
        """String."""
        return str(self.content)


class BaseTool:
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        pass

    @abstractmethod
    def __call__(self, input: Any) -> ToolOutput:
        pass

    def _process_langchain_tool_kwargs(
        self,
        langchain_tool_kwargs: Any,
    ) -> Dict[str, Any]:
        """Process langchain tool kwargs."""
        if "name" not in langchain_tool_kwargs:
            langchain_tool_kwargs["name"] = self.metadata.name or ""
        if "description" not in langchain_tool_kwargs:
            langchain_tool_kwargs["description"] = self.metadata.description
        if "fn_schema" not in langchain_tool_kwargs:
            langchain_tool_kwargs["args_schema"] = self.metadata.fn_schema
        return langchain_tool_kwargs

    def to_langchain_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> Tool:
        """To langchain tool."""
        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return Tool.from_function(
            func=self.__call__,
            **langchain_tool_kwargs,
        )

    def to_langchain_structured_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> StructuredTool:
        """To langchain structured tool."""
        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return StructuredTool.from_function(
            func=self.__call__,
            **langchain_tool_kwargs,
        )


class AsyncBaseTool(BaseTool):
    """
    Base-level tool class that is backwards compatible with the old tool spec but also
    supports async.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, input: Any) -> ToolOutput:
        """
        This is the method that should be implemented by the tool developer.
        """

    @abstractmethod
    async def acall(self, input: Any) -> ToolOutput:
        """
        This is the async version of the call method.
        Should also be implemented by the tool developer as an
        async-compatible implementation.
        """


class BaseToolAsyncAdapter(AsyncBaseTool):
    """
    Adapter class that allows a synchronous tool to be used as an async tool.
    """

    def __init__(self, tool: BaseTool):
        self.base_tool = tool

    @property
    def metadata(self) -> ToolMetadata:
        return self.base_tool.metadata

    def call(self, input: Any) -> ToolOutput:
        return self.base_tool(input)

    async def acall(self, input: Any) -> ToolOutput:
        return self.call(input)


def adapt_to_async_tool(tool: BaseTool) -> AsyncBaseTool:
    """
    Converts a synchronous tool to an async tool.
    """
    if isinstance(tool, AsyncBaseTool):
        return tool
    else:
        return BaseToolAsyncAdapter(tool)
