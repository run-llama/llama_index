import inspect
import json
from typing import TYPE_CHECKING, Any, Sequence

from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import (
    AsyncBaseTool,
    BaseTool,
    ToolOutput,
    adapt_to_async_tool,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


def _function_tool_accepts_kwargs(tool: FunctionTool, arguments: dict) -> bool:
    """Check whether a function tool can receive its generated arguments as kwargs."""
    try:
        signature = inspect.signature(tool.real_fn)
    except (TypeError, ValueError):
        return True

    try:
        signature.bind_partial(**arguments)
    except TypeError:
        return False
    return True


def _tool_entrypoint(tool: BaseTool, async_call: bool) -> Any:
    """Return the callable that actually receives the tool arguments."""
    if isinstance(tool, AsyncBaseTool):
        return tool.acall if async_call else tool.call
    # plain BaseTool: invoked through __call__, also by the async adapter
    return tool


def _entrypoint_needs_kwargs(fn: Any, single_arg: Any) -> bool:
    """
    Select a call shape before executing, without retrying tool failures.

    Preserve positional dispatch when it binds, including permissive *args
    tools. Binding also detects required keyword-only parameters even when
    *args is present. Otherwise use kwargs, as for multi-argument tools and
    entry points whose signatures cannot be inspected.
    """
    try:
        signature = inspect.signature(fn)
        signature.bind(single_arg)
    except (TypeError, ValueError):
        return True
    return False


def call_tool(tool: BaseTool, arguments: dict) -> ToolOutput:
    """Call a tool with arguments."""
    from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool

    try:
        # The loader requires its configurable query key in kwargs, even
        # though it also supports direct positional reader arguments.
        if isinstance(tool, OnDemandLoaderTool):
            return tool(**arguments)
        if (
            len(tool.metadata.get_parameters_dict()["properties"]) == 1
            and len(arguments) == 1
        ):
            single_arg = arguments[next(iter(arguments))]
            if isinstance(tool, FunctionTool):
                if _function_tool_accepts_kwargs(tool, arguments):
                    return tool(**arguments)
                return tool(single_arg)
            if _entrypoint_needs_kwargs(
                _tool_entrypoint(tool, async_call=False), single_arg
            ):
                return tool(**arguments)
            return tool(single_arg)
        else:
            return tool(**arguments)
    except Exception as e:
        return ToolOutput(
            content="Encountered error: " + str(e),
            tool_name=tool.metadata.get_name(),
            raw_input=arguments,
            raw_output=str(e),
            is_error=True,
            exception=e,
        )


async def acall_tool(tool: BaseTool, arguments: dict) -> ToolOutput:
    """Call a tool with arguments asynchronously."""
    from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool

    async_tool = adapt_to_async_tool(tool)
    try:
        if isinstance(tool, OnDemandLoaderTool):
            return await async_tool.acall(**arguments)
        if (
            len(tool.metadata.get_parameters_dict()["properties"]) == 1
            and len(arguments) == 1
        ):
            single_arg = arguments[next(iter(arguments))]
            if isinstance(tool, FunctionTool):
                if _function_tool_accepts_kwargs(tool, arguments):
                    return await async_tool.acall(**arguments)
                return await async_tool.acall(single_arg)
            if _entrypoint_needs_kwargs(
                _tool_entrypoint(tool, async_call=True), single_arg
            ):
                return await async_tool.acall(**arguments)
            return await async_tool.acall(single_arg)
        else:
            return await async_tool.acall(**arguments)
    except Exception as e:
        return ToolOutput(
            content="Encountered error: " + str(e),
            tool_name=tool.metadata.get_name(),
            raw_input=arguments,
            raw_output=str(e),
            is_error=True,
            exception=e,
        )


def call_tool_with_selection(
    tool_call: ToolSelection,
    tools: Sequence["BaseTool"],
    verbose: bool = False,
) -> ToolOutput:
    from llama_index.core.tools.calling import call_tool

    tools_by_name = {tool.metadata.name: tool for tool in tools}
    name = tool_call.tool_name
    if verbose:
        arguments_str = json.dumps(tool_call.tool_kwargs)
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = tools_by_name[name]
    output = call_tool(tool, tool_call.tool_kwargs)

    if verbose:
        print("=== Function Output ===")
        print(output.content)

    return output


async def acall_tool_with_selection(
    tool_call: ToolSelection,
    tools: Sequence["BaseTool"],
    verbose: bool = False,
) -> ToolOutput:
    from llama_index.core.tools.calling import acall_tool

    tools_by_name = {tool.metadata.name: tool for tool in tools}
    name = tool_call.tool_name
    if verbose:
        arguments_str = json.dumps(tool_call.tool_kwargs)
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = tools_by_name[name]
    output = await acall_tool(tool, tool_call.tool_kwargs)

    if verbose:
        print("=== Function Output ===")
        print(output.content)

    return output
