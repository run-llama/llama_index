import asyncio
import contextvars
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
    Tuple,
    get_origin,
)
import re

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
    DocumentBlock,
    VideoBlock,
)
from llama_index.core.bridge.pydantic import BaseModel, FieldInfo
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.schema import BaseNode, Document
from llama_index.core.workflow.context import Context

AsyncCallable = Callable[..., Awaitable[Any]]


def _is_context_param(param_annotation: Any) -> bool:
    """Check if a parameter annotation is Context or Context[SomeType]."""
    return param_annotation == Context or (get_origin(param_annotation) is Context)


def sync_to_async(fn: Callable[..., Any]) -> AsyncCallable:
    """Sync to async."""

    async def _async_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(None, lambda: ctx.run(fn, *args, **kwargs))

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
        pre_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        async_pre_processor: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        post_processor: Optional[Callable[[ToolOutput], ToolOutput]] = None,
        async_post_processor: Optional[Callable[[ToolOutput], Awaitable[ToolOutput]]] = None,
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

        # Handle pre-processor (sync and async)
        self._pre_processor = None
        if pre_processor is not None:
            self._pre_processor = pre_processor
        elif async_pre_processor is not None:
            self._pre_processor = async_to_sync(async_pre_processor)

        self._async_pre_processor = None
        if async_pre_processor is not None:
            self._async_pre_processor = async_pre_processor
        elif self._pre_processor is not None:
            self._async_pre_processor = sync_to_async(self._pre_processor)

        # Handle post-processor (sync and async)
        self._post_processor = None
        if post_processor is not None:
            self._post_processor = post_processor
        elif async_post_processor is not None:
            self._post_processor = async_to_sync(async_post_processor)

        self._async_post_processor = None
        if async_post_processor is not None:
            self._async_post_processor = async_post_processor
        elif self._post_processor is not None:
            self._async_post_processor = sync_to_async(self._post_processor)

        self.partial_params = partial_params or {}

        # Extract actual default values from FieldInfo defaults so they are
        # applied when the function is called without those arguments.
        self._field_defaults: Dict[str, Any] = {}
        for param in sig.parameters.values():
            if isinstance(param.default, FieldInfo) and not param.default.is_required():
                self._field_defaults[param.name] = param.default.get_default(
                    call_default_factory=True
                )

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

    def _run_sync_pre_processor(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the sync pre-processor, if provided."""
        if self._pre_processor:
            return self._pre_processor(arguments)
        return arguments

    async def _run_async_pre_processor(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the async pre-processor, if provided."""
        if self._async_pre_processor:
            return await self._async_pre_processor(arguments)
        return arguments

    def _run_sync_post_processor(self, result: ToolOutput) -> ToolOutput:
        """Runs the sync post-processor, if provided."""
        if self._post_processor:
            return self._post_processor(result)
        return result

    async def _run_async_post_processor(self, result: ToolOutput) -> ToolOutput:
        """Runs the async post-processor, if provided."""
        if self._async_post_processor:
            return await self._async_post_processor(result)
        return result

    def __call__(self, input: Any) -> ToolOutput:
        """
        Call the tool with input.
        
        This method handles both pre-processing of inputs and post-processing of outputs
        in a deterministic way before and after execution.
        """
        # Convert input to dict if needed (for compatibility with existing code)
        if isinstance(input, str):
            arguments = {"input": input}
        elif isinstance(input, dict):
            arguments = input
        else:
            arguments = {"input": input}

        # Apply pre-processor if provided
        arguments = self._run_sync_pre_processor(arguments)

        # Call the actual function
        try:
            result = self._fn(**arguments)
            if inspect.iscoroutine(result):
                # This shouldn't happen in sync call, but handle gracefully
                raise ValueError("Unexpected async result in sync tool call")
        except Exception as e:
            return ToolOutput(
                content="Encountered error: " + str(e),
                tool_name=self.metadata.get_name(),
                raw_input=arguments,
                raw_output=str(e),
                is_error=True,
                exception=e,
            )

        # Apply callback if provided (post-processing)
        callback_result = self._run_sync_callback(result)
        if callback_result is not None:
            if isinstance(callback_result, ToolOutput):
                result = callback_result
            else:
                # Override the content only
                result = ToolOutput(
                    tool_name=self.metadata.get_name(),
                    content=callback_result,
                    raw_input=arguments,
                    raw_output=result,
                )
        
        # Apply post-processor if provided
        result = self._run_sync_post_processor(result)
        
        return result

    async def acall(self, input: Any) -> ToolOutput:
        """
        Async call the tool with input.
        
        This method handles both pre-processing of inputs and post-processing of outputs
        in a deterministic way before and after execution.
        """
        # Convert input to dict if needed (for compatibility with existing code)
        if isinstance(input, str):
            arguments = {"input": input}
        elif isinstance(input, dict):
            arguments = input
        else:
            arguments = {"input": input}

        # Apply pre-processor if provided
        arguments = await self._run_async_pre_processor(arguments)

        # Call the actual function
        try:
            result = await self._async_fn(**arguments)
        except Exception as e:
            return ToolOutput(
                content="Encountered error: " + str(e),
                tool_name=self.metadata.get_name(),
                raw_input=arguments,
                raw_output=str(e),
                is_error=True,
                exception=e,
            )

        # Apply callback if provided (post-processing)
        callback_result = await self._run_async_callback(result)
        if callback_result is not None:
            if isinstance(callback_result, ToolOutput):
                result = callback_result
            else:
                # Override the content only
                result = ToolOutput(
                    tool_name=self.metadata.get_name(),
                    content=callback_result,
                    raw_input=arguments,
                    raw_output=result,
                )
        
        # Apply post-processor if provided
        result = await self._run_async_post_processor(result)
        
        return result

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
        pre_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        async_pre_processor: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        post_processor: Optional[Callable[[ToolOutput], ToolOutput]] = None,
        async_post_processor: Optional[Callable[[ToolOutput], Awaitable[ToolOutput]]] = None,
    ) -> "FunctionTool":
        partial_params = partial_params or {}

        if tool_metadata is None:
            fn_to_parse = fn or async_fn
            assert fn_to_parse is not None, "fn must be provided"
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__ or ""

            # Get function signature
            fn_sig = inspect.signature(fn_to_parse)
            fn_params = set(fn_sig.parameters.keys())

            # 1. Extract docstring param descriptions
            param_docs, unknown_params = cls.extract_param_docs(docstring, fn_params)

            # 2. Filter context and self in a single pass
            ctx_param_name = None
            has_self = False
            filtered_params = []
            for param in fn_sig.parameters.values():
                if _is_context_param(param.annotation):
                    ctx_param_name = param.name
                    continue
                if param.name == "self":
                    has_self = True
                    continue
                filtered_params.append(param)

            # 3. Remove FieldInfo defaults and partial_params
            final_params = [
                param.replace(default=inspect.Parameter.empty)
                if isinstance(param.default, FieldInfo)
                else param
                for param in filtered_params
                if param.name not in (partial_params or {})
            ]

            # 4. Replace signature in one go
            fn_sig = fn_sig.replace(parameters=final_params)

            # 5. Build description
            if description is None:
                description = f"{name}{fn_sig}\n"
                if docstring:
                    description += docstring

                description = description.strip()

            # 6. Get function schema
            if fn_schema is None:
                try:
                    fn_schema = create_schema_from_function(
                        fn_to_parse, name=name, description=description
                    )
                except Exception:
                    pass

            # 7. Create metadata
            tool_metadata = ToolMetadata(
                name=name,
                description=description,
                fn_schema=fn_schema,
                return_direct=return_direct,
            )

        # Create the FunctionTool with pre- and post-processors
        return cls(
            fn=fn,
            async_fn=async_fn,
            metadata=tool_metadata,
            callback=callback,
            async_callback=async_callback,
            partial_params=partial_params,
            pre_processor=pre_processor,
            async_pre_processor=async_pre_processor,
            post_processor=post_processor,
            async_post_processor=async_post_processor,
        )

    def call(self, input: Any) -> ToolOutput:
        """Synchronous tool call."""
        return self.__call__(input)

    async def acall_async(self, input: Any) -> ToolOutput:
        """Async tool call."""
        return await self.acall(input)
