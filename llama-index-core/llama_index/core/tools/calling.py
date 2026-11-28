import inspect
import json
from typing import TYPE_CHECKING, Sequence

from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import BaseTool, ToolOutput, adapt_to_async_tool

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


def call_tool(tool: BaseTool, arguments: dict) -> ToolOutput:
    """Call a tool with arguments."""
    try:
        if (
            len(tool.metadata.get_parameters_dict()["properties"]) == 1
            and len(arguments) == 1
        ):
            single_arg = arguments[next(iter(arguments))]
            if isinstance(tool, FunctionTool):
                if _function_tool_accepts_kwargs(tool, arguments):
                    return tool(**arguments)
                return tool(single_arg)
            try:
                return tool(single_arg)
            except Exception:
                # some tools will REQUIRE kwargs, so try it
                return tool(**arguments)
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
    async_tool = adapt_to_async_tool(tool)
    try:
        if (
            len(tool.metadata.get_parameters_dict()["properties"]) == 1
            and len(arguments) == 1
        ):
            single_arg = arguments[next(iter(arguments))]
            if isinstance(tool, FunctionTool):
                if _function_tool_accepts_kwargs(tool, arguments):
                    return await async_tool.acall(**arguments)
                return await async_tool.acall(single_arg)
            try:
                return await async_tool.acall(single_arg)
            except Exception:
                # some tools will REQUIRE kwargs, so try it
                return await async_tool.acall(**arguments)
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


def _tool_not_found_output(tool_call: ToolSelection) -> ToolOutput:
    return ToolOutput(
        content=f"Tool {tool_call.tool_name} not found. Please select a tool that is available.",
        tool_name=tool_call.tool_name,
        raw_input=tool_call.tool_kwargs,
        raw_output=None,
        is_error=True,
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
    if name not in tools_by_name:
        output = _tool_not_found_output(tool_call)
    else:
        output = call_tool(tools_by_name[name], tool_call.tool_kwargs)

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
    if name not in tools_by_name:
        output = _tool_not_found_output(tool_call)
    else:
        output = await acall_tool(tools_by_name[name], tool_call.tool_kwargs)

    if verbose:
        print("=== Function Output ===")
        print(output.content)

    return output
