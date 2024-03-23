from llama_index.core.tools.types import BaseTool, ToolOutput, adapt_to_async_tool


def call_tool(tool: BaseTool, arguments: dict) -> ToolOutput:
    """Call a tool with arguments."""
    try:
        if (
            len(tool.metadata.get_parameters_dict()["properties"]) == 1
            and len(arguments) == 1
        ):
            single_arg = arguments[next(iter(arguments))]
            return tool(single_arg)
        else:
            return tool(**arguments)
    except Exception as e:
        return ToolOutput(
            content="Encountered error: " + str(e),
            tool_name=tool.metadata.name,
            raw_input=arguments,
            raw_output=str(e),
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
            return await async_tool.acall(single_arg)
        else:
            return await async_tool.acall(**arguments)
    except Exception as e:
        return ToolOutput(
            content="Encountered error: " + str(e),
            tool_name=tool.metadata.name,
            raw_input=arguments,
            raw_output=str(e),
        )
