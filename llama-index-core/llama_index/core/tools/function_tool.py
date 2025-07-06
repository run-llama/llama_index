import asyncio
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_origin,
)


if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import StructuredTool, Tool

from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.llms.types import (
    TextBlock,
    ImageBlock,
    AudioBlock,
    CitableBlock,
    CitationBlock,
    ContentBlock,
)
from llama_index.core.bridge.pydantic import BaseModel, FieldInfo
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.workflow.context import Context

AsyncCallable = Callable[..., Awaitable[Any]]


def _is_context_param(param_annotation: Any) -> bool:
    """Check if a parameter annotation is Context or Context[SomeType]."""
    return param_annotation == Context or (get_origin(param_annotation) is Context)


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


# The type that the callback can return: either a ToolOutput instance or a string to override the content.
CallbackReturn = Optional[Union[ToolOutput, str]]


class FunctionTool(AsyncBaseTool):
    """
    Function Tool.

    A tool that takes in a function, optionally handles workflow context,
    and allows the use of callbacks. The callback can return a new ToolOutput
    to override the default one or a string that will be used as the final content.
    """

    def __init__(
        self,
        fn: Optional[Callable[..., Any]] = None,
        metadata: Optional[ToolMetadata] = None,
        async_fn: Optional[AsyncCallable] = None,
        callback: Optional[Callable[..., Any]] = None,
        async_callback: Optional[Callable[..., Any]] = None,
        partial_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if fn is None and async_fn is None:
            raise ValueError("fn or async_fn must be provided.")

        # Handle function (sync and async)
        self._real_fn = fn or async_fn
        if async_fn is not None:
            self._async_fn = async_fn
            self._fn = fn or async_to_sync(async_fn)
        else:
            assert fn is not None
            if inspect.iscoroutinefunction(fn):
                self._async_fn = fn
                self._fn = async_to_sync(fn)
            else:
                self._fn = fn
                self._async_fn = sync_to_async(fn)

        # Determine if the function requires context by inspecting its signature
        fn_to_inspect = fn or async_fn
        assert fn_to_inspect is not None
        sig = inspect.signature(fn_to_inspect)
        self.requires_context = any(
            _is_context_param(param.annotation) for param in sig.parameters.values()
        )
        self.ctx_param_name = (
            next(
                param.name
                for param in sig.parameters.values()
                if _is_context_param(param.annotation)
            )
            if self.requires_context
            else None
        )

        if metadata is None:
            raise ValueError("metadata must be provided")
        self._metadata = metadata

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

        self.partial_params = partial_params or {}

    def _run_sync_callback(self, result: Any) -> CallbackReturn:
        """
        Runs the sync callback, if provided, and returns either a ToolOutput
        to override the default output or a string to override the content.
        """
        if self._callback:
            ret: CallbackReturn = self._callback(result)
            return ret
        return None

    async def _run_async_callback(self, result: Any) -> CallbackReturn:
        """
        Runs the async callback, if provided, and returns either a ToolOutput
        to override the default output or a string to override the content.
        """
        if self._async_callback:
            ret: CallbackReturn = await self._async_callback(result)
            return ret
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
        partial_params: Optional[Dict[str, Any]] = None,
    ) -> "FunctionTool":
        partial_params = partial_params or {}

        if tool_metadata is None:
            fn_to_parse = fn or async_fn
            assert fn_to_parse is not None, "fn must be provided"
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__ or ""

            # Get function signature
            fn_sig = inspect.signature(fn_to_parse)

            # Remove ctx parameter from schema if present
            ctx_param_name = None
            for param in fn_sig.parameters.values():
                if _is_context_param(param.annotation):
                    ctx_param_name = param.name
                    fn_sig = fn_sig.replace(
                        parameters=[
                            param
                            for param in fn_sig.parameters.values()
                            if not _is_context_param(param.annotation)
                        ]
                    )

            # Remove self parameter from schema if present
            has_self = False
            for param in fn_sig.parameters.values():
                if param.name == "self":
                    has_self = True
                    fn_sig = fn_sig.replace(
                        parameters=[
                            param
                            for param in fn_sig.parameters.values()
                            if param.name != "self"
                        ]
                    )
                    break

            # Handle FieldInfo defaults
            fn_sig = fn_sig.replace(
                parameters=[
                    param.replace(default=inspect.Parameter.empty)
                    if isinstance(param.default, FieldInfo)
                    else param
                    for param in fn_sig.parameters.values()
                    if param.name not in partial_params
                ]
            )

            description = description or f"{name}{fn_sig}\n{docstring}"
            if fn_schema is None:
                ignore_fields = []
                if ctx_param_name is not None:
                    ignore_fields.append(ctx_param_name)
                if has_self:
                    ignore_fields.append("self")
                ignore_fields.extend(partial_params.keys())

                fn_schema = create_schema_from_function(
                    f"{name}",
                    fn_to_parse,
                    additional_fields=None,
                    ignore_fields=ignore_fields,
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
            partial_params=partial_params,
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

    @property
    def real_fn(self) -> Union[Callable[..., Any], AsyncCallable]:
        """Real function."""
        if self._real_fn is None:
            raise ValueError("Real function is not set!")

        return self._real_fn

    def _parse_tool_output(self, raw_output: Any) -> List[ContentBlock]:
        """Parse tool output into content blocks."""
        if isinstance(
            raw_output, (TextBlock, ImageBlock, AudioBlock, CitableBlock, CitationBlock)
        ):
            return [raw_output]
        elif isinstance(raw_output, list) and all(
            isinstance(
                item, (TextBlock, ImageBlock, AudioBlock, CitableBlock, CitationBlock)
            )
            for item in raw_output
        ):
            return raw_output
        else:
            return [TextBlock(text=str(raw_output))]

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        all_kwargs = {**self.partial_params, **kwargs}
        return self.call(*args, **all_kwargs)

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Sync Call."""
        all_kwargs = {**self.partial_params, **kwargs}
        if self.requires_context and self.ctx_param_name is not None:
            if self.ctx_param_name not in all_kwargs:
                raise ValueError("Context is required for this tool")

        raw_output = self._fn(*args, **all_kwargs)

        # Exclude the Context param from the tool output so that the Context can be serialized
        tool_output_kwargs = {
            k: v for k, v in all_kwargs.items() if k != self.ctx_param_name
        }

        # Parse tool output into content blocks
        output_blocks = self._parse_tool_output(raw_output)

        # Default ToolOutput based on the raw output
        default_output = ToolOutput(
            blocks=output_blocks,
            tool_name=self.metadata.get_name(),
            raw_input={"args": args, "kwargs": tool_output_kwargs},
            raw_output=raw_output,
        )
        # Check for a sync callback override
        callback_result = self._run_sync_callback(raw_output)
        if callback_result is not None:
            if isinstance(callback_result, ToolOutput):
                return callback_result
            else:
                # Assume callback_result is a string to override the content.
                return ToolOutput(
                    content=str(callback_result),
                    tool_name=self.metadata.get_name(),
                    raw_input={"args": args, "kwargs": tool_output_kwargs},
                    raw_output=raw_output,
                )
        return default_output

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Async Call."""
        all_kwargs = {**self.partial_params, **kwargs}
        if self.requires_context and self.ctx_param_name is not None:
            if self.ctx_param_name not in all_kwargs:
                raise ValueError("Context is required for this tool")

        raw_output = await self._async_fn(*args, **all_kwargs)

        # Exclude the Context param from the tool output so that the Context can be serialized
        tool_output_kwargs = {
            k: v for k, v in all_kwargs.items() if k != self.ctx_param_name
        }

        # Parse tool output into content blocks
        output_blocks = self._parse_tool_output(raw_output)

        # Default ToolOutput based on the raw output
        default_output = ToolOutput(
            blocks=output_blocks,
            tool_name=self.metadata.get_name(),
            raw_input={"args": args, "kwargs": tool_output_kwargs},
            raw_output=raw_output,
        )
        # Check for an async callback override
        callback_result = await self._run_async_callback(raw_output)
        if callback_result is not None:
            if isinstance(callback_result, ToolOutput):
                return callback_result
            else:
                # Assume callback_result is a string to override the content.
                return ToolOutput(
                    content=str(callback_result),
                    tool_name=self.metadata.get_name(),
                    raw_input={"args": args, "kwargs": tool_output_kwargs},
                    raw_output=raw_output,
                )
        return default_output

    def to_langchain_tool(self, **langchain_tool_kwargs: Any) -> "Tool":
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
        self, **langchain_tool_kwargs: Any
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
