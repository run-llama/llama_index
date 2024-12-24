import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Type

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import StructuredTool, Tool

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import BaseModel, FieldInfo
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.core.tools.utils import create_schema_from_function

AsyncCallable = Callable[..., Awaitable[Any]]


def sync_to_async(fn: Callable[..., Any]) -> AsyncCallable:
    """Sync to async."""

    async def _async_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    return _async_wrapped_fn


def async_to_sync(func_async: AsyncCallable) -> Callable:
    """Async to sync."""

    def _sync_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        return asyncio_run(func_async(*args, **kwargs))  # type: ignore[arg-type]

    return _sync_wrapped_fn


class FunctionTool(AsyncBaseTool):
    """Function Tool.

    A tool that takes in a function and a callback.

    """

    def __init__(
        self,
        fn: Optional[Callable[..., Any]] = None,
        metadata: Optional[ToolMetadata] = None,
        async_fn: Optional[Callable[..., Any]] = None,
        callback: Optional[Callable[..., Any]] = None,
        async_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        if fn is None and async_fn is None:
            raise ValueError("fn or async_fn must be provided.")

        # Handle function (sync and async)
        if fn is not None:
            self._fn = fn
        elif async_fn is not None:
            self._fn = async_to_sync(async_fn)

        if async_fn is not None:
            self._async_fn = async_fn
        elif fn is not None:
            self._async_fn = sync_to_async(self._fn)

        if metadata is None:
            raise ValueError("metadata must be provided.")

        # Handle callback (sync and async)
        self._callback = None
        if callback is not None:
            self._callback = callback
        elif async_callback is not None:
            self._callback = async_to_sync(async_callback)

        self._async_callback = None
        if async_callback is not None:
            self._async_callback = async_callback
        elif self._callback is not None:
            self._async_callback = sync_to_async(self._callback)

        self._metadata = metadata

    def _run_sync_callback(self, result: Any) -> Any:
        """Runs the sync callback, if provided."""
        if self._callback:
            return self._callback(result)
        return None

    async def _run_async_callback(self, result: Any) -> Any:
        """Runs the async callback, if provided."""
        if self._async_callback:
            return await self._async_callback(result)
        return None

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
        callback: Optional[Callable[[Any], Any]] = None,
        async_callback: Optional[AsyncCallable] = None,
    ) -> "FunctionTool":
        if tool_metadata is None:
            fn_to_parse = fn or async_fn
            assert fn_to_parse is not None, "fn or async_fn must be provided."
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__

            # Make a new function signature with FieldInfo defaults removed.
            # The information in FieldInfo is covered by fn_schema.
            fn_sig = inspect.signature(fn_to_parse)
            fn_sig = fn_sig.replace(
                parameters=[
                    param.replace(default=inspect.Parameter.empty)
                    if isinstance(param.default, FieldInfo)
                    else param
                    for param in fn_sig.parameters.values()
                ]
            )

            description = description or f"{name}{fn_sig}\n{docstring}"
            if fn_schema is None:
                fn_schema = create_schema_from_function(
                    f"{name}", fn_to_parse, additional_fields=None
                )
            tool_metadata = ToolMetadata(
                name=name,
                description=description,
                fn_schema=fn_schema,
                return_direct=return_direct,
            )
        return cls(
            fn=fn,
            metadata=tool_metadata,
            async_fn=async_fn,
            callback=callback,
            async_callback=async_callback,
        )

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
        """Sync Call."""
        tool_output = self._fn(*args, **kwargs)
        final_output_content = str(tool_output)
        # Execute sync callback, if available
        callback_output = self._run_sync_callback(tool_output)
        if callback_output:
            final_output_content += f" Callback: {callback_output}"
        return ToolOutput(
            content=final_output_content,
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Async Call."""
        tool_output = await self._async_fn(*args, **kwargs)
        final_output_content = str(tool_output)
        # Execute async callback, if available
        callback_output = await self._run_async_callback(tool_output)
        if callback_output:
            final_output_content += f" Callback: {callback_output}"
        return ToolOutput(
            content=final_output_content,
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    def to_langchain_tool(
        self,
        **langchain_tool_kwargs: Any,
    ) -> "Tool":
        """To langchain tool."""
        from llama_index.core.bridge.langchain import Tool

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
    ) -> "StructuredTool":
        """To langchain structured tool."""
        from llama_index.core.bridge.langchain import StructuredTool

        langchain_tool_kwargs = self._process_langchain_tool_kwargs(
            langchain_tool_kwargs
        )
        return StructuredTool.from_function(
            func=self.fn,
            coroutine=self.async_fn,
            **langchain_tool_kwargs,
        )
