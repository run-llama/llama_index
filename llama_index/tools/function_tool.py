from typing import Any, Optional, Callable

from llama_index.tools.types import BaseTool, ToolMetadata
from langchain.tools import Tool, StructuredTool
from inspect import signature

DEFAULT_NAME = "Function Tool"
DEFAULT_DESCRIPTION = """Useful for running a natural language query
against a function and getting back a response.

"""


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
    ) -> "FunctionTool":
        name = name or fn.__name__
        docstring = fn.__doc__
        description = description or f"{name}{signature(fn)}\n{docstring}"
        metadata = ToolMetadata(name=name, description=description)
        return cls(fn=fn, metadata=metadata)

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        """Function."""
        return self._fn

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Call."""
        return self._fn(*args, **kwargs)

    def to_langchain_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> Tool:
        """To langchain tool."""
        return Tool.from_function(
            fn=self.fn,
            name=self.metadata.name or "",
            description=self.metadata.description,
            **langchain_tool_kwargs,
        )

    def to_langchain_structured_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> StructuredTool:
        """To langchain structured tool."""
        return StructuredTool.from_function(
            fn=self.fn,
            name=self.metadata.name,
            description=self.metadata.description,
            **langchain_tool_kwargs,
        )
