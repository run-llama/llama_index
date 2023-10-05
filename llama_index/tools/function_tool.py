from inspect import signature
from typing import Any, Awaitable, Callable, Optional, Type

from llama_index.bridge.langchain import StructuredTool, Tool
from llama_index.bridge.pydantic import BaseModel
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.tools.utils import create_schema_from_function

AsyncCallable = Callable[..., Awaitable[Any]]


def sync_to_async(fn: Callable[..., Any]) -> AsyncCallable:
    """Sync to async."""

    async def _async_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return _async_wrapped_fn


class FunctionTool(AsyncBaseTool):
    """Function Tool.

    A tool that takes in a function.

    """

    def __init__(
        self,
        fn: Callable[..., Any],
        metadata: ToolMetadata,
        async_fn: Optional[AsyncCallable] = None,
    ) -> None:
        self._fn = fn
        if async_fn is not None:
            self._async_fn = async_fn
        else:
            self._async_fn = sync_to_async(self._fn)
        self._metadata = metadata

    @classmethod
    def from_defaults(
        cls,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
    ) -> "FunctionTool":
        if tool_metadata is None:
            name = name or fn.__name__
            docstring = fn.__doc__
            description = description or f"{name}{signature(fn)}\n{docstring}"
            if fn_schema is None:
                fn_schema = create_schema_from_function(
                    f"{name}", fn, additional_fields=None
                )
            tool_metadata = ToolMetadata(
                name=name, description=description, fn_schema=fn_schema
            )
        return cls(fn=fn, metadata=tool_metadata, async_fn=async_fn)

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        """Function."""
        return self._fn

    @property
    def async_fn(self) -> AsyncCallable:
        """Async function."""
        return self._async_fn

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        tool_output = self._fn(*args, **kwargs)
        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        tool_output = await self._async_fn(*args, **kwargs)
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
            coroutine=self.async_fn,
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
            coroutine=self.async_fn,
            **langchain_tool_kwargs,
        )
