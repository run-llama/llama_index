from typing import Any, Optional, Callable, Type

from pydantic import BaseModel
from llama_index.tools.types import BaseTool, ToolMetadata, ToolOutput
from llama_index.bridge.langchain import Tool, StructuredTool
from inspect import signature
from llama_index.tools.utils import create_schema_from_function


class FunctionTool(BaseTool):
    """Function Tool.

    A tool that takes in a function.

    """

    def __init__(
        self,
        fn: Callable[..., Any],
        metadata: ToolMetadata,
    ) -> None:
        self._fn = fn
        self._metadata = metadata

    @classmethod
    def from_defaults(
        cls,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
    ) -> "FunctionTool":
        name = name or fn.__name__
        docstring = fn.__doc__
        description = description or f"{name}{signature(fn)}\n{docstring}"
        if fn_schema is None:
            fn_schema = create_schema_from_function(
                f"{name}", fn, additional_fields=None
            )
        metadata = ToolMetadata(name=name, description=description, fn_schema=fn_schema)
        return cls(fn=fn, metadata=metadata)

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        """Function."""
        return self._fn

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        tool_output = self._fn(*args, **kwargs)
        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    def to_langchain_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> Tool:
        """To langchain tool."""
        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return Tool.from_function(
            func=self.fn,
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
            func=self.fn,
            **langchain_tool_kwargs,
        )
