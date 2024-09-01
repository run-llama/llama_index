from llama_index.core.tools.types import BaseTool, ToolOutput, adapt_to_async_tool
from typing import TYPE_CHECKING, List
from llama_index.core.llms.llm import ToolSelection
import json

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


def call_tool(tool: BaseTool, arguments: dict) -> ToolOutput:
    """Call a tool with arguments."""
    try:
        if (
            len(tool.metadata.get_parameters_dict()["properties"]) == 1
            and len(arguments) == 1
        ):
            try:
                single_arg = arguments[next(iter(arguments))]
                return tool(single_arg)
            except Exception:
                # some tools will REQUIRE kwargs, so try it
                return tool(**arguments)
        else:
            return tool(**arguments)
    except Exception as e:
        return ToolOutput(
            content="Encountered error: " + str(e),
            tool_name=tool.metadata.name,
            raw_input=arguments,
            raw_output=str(e),
            is_error=True,
        )


async def acall_tool(tool: BaseTool, arguments: dict) -> ToolOutput:
    """Call a tool with arguments asynchronously."""
    async_tool = adapt_to_async_tool(tool)
    try:
        if (
            len(tool.metadata.get_parameters_dict()["properties"]) == 1
            and len(arguments) == 1
        ):
            try:
                single_arg = arguments[next(iter(arguments))]
                return await async_tool.acall(single_arg)
            except Exception:
                # some tools will REQUIRE kwargs, so try it
                return await async_tool.acall(**arguments)
        else:
            return await async_tool.acall(**arguments)
    except Exception as e:
        return ToolOutput(
            content="Encountered error: " + str(e),
            tool_name=tool.metadata.name,
            raw_input=arguments,
            raw_output=str(e),
            is_error=True,
        )


def call_tool_with_selection(
    tool_call: ToolSelection,
    tools: List["BaseTool"],
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
    tools: List["BaseTool"],
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
