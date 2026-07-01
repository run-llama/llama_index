import json
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from llama_index.core.tools.types import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload

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
            tool_name=tool.metadata.get_name(),
            raw_input=arguments,
            raw_output=str(e),
            is_error=True,
            exception=e,
        )


def _tool_audit_payload(
    *,
    tool_name: str,
    tool_kwargs: Dict,
    tool_id: Optional[str],
    output: Optional[ToolOutput] = None,
) -> Dict:
    payload = {
        "tool_name": tool_name,
        "tool_kwargs": tool_kwargs,
        "tool_id": tool_id,
        "is_error": False,
        "output": None,
        "error": None,
    }

    if output is not None:
        payload["is_error"] = output.is_error
        payload["output"] = output.content
        if output.is_error:
            payload["error"] = (
                str(output.exception)
                if output.exception is not None
                else str(output.raw_output)
            )

    return {EventPayload.TOOL: payload}


def call_tool_with_selection(
    tool_call: ToolSelection,
    tools: Sequence["BaseTool"],
    verbose: bool = False,
    *,
    callback_manager: Optional[CallbackManager] = None,
    tool_id: Optional[str] = None,
) -> ToolOutput:
    from llama_index.core.tools.calling import call_tool

    tools_by_name = {tool.metadata.name: tool for tool in tools}
    name = tool_call.tool_name
    if verbose:
        arguments_str = json.dumps(tool_call.tool_kwargs)
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = tools_by_name[name]
    if callback_manager is not None:
        with callback_manager.event(
            CBEventType.TOOL,
            payload=_tool_audit_payload(
                tool_name=name,
                tool_kwargs=tool_call.tool_kwargs,
                tool_id=tool_id or tool_call.tool_id,
            ),
        ) as event:
            output = call_tool(tool, tool_call.tool_kwargs)
            event.on_end(
                payload=_tool_audit_payload(
                    tool_name=name,
                    tool_kwargs=tool_call.tool_kwargs,
                    tool_id=tool_id or tool_call.tool_id,
                    output=output,
                )
            )
    else:
        output = call_tool(tool, tool_call.tool_kwargs)

    if verbose:
        print("=== Function Output ===")
        print(output.content)

    return output


async def acall_tool_with_selection(
    tool_call: ToolSelection,
    tools: Sequence["BaseTool"],
    verbose: bool = False,
    *,
    callback_manager: Optional[CallbackManager] = None,
    tool_id: Optional[str] = None,
) -> ToolOutput:
    from llama_index.core.tools.calling import acall_tool

    tools_by_name = {tool.metadata.name: tool for tool in tools}
    name = tool_call.tool_name
    if verbose:
        arguments_str = json.dumps(tool_call.tool_kwargs)
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = tools_by_name[name]
    if callback_manager is not None:
        with callback_manager.event(
            CBEventType.TOOL,
            payload=_tool_audit_payload(
                tool_name=name,
                tool_kwargs=tool_call.tool_kwargs,
                tool_id=tool_id or tool_call.tool_id,
            ),
        ) as event:
            output = await acall_tool(tool, tool_call.tool_kwargs)
            event.on_end(
                payload=_tool_audit_payload(
                    tool_name=name,
                    tool_kwargs=tool_call.tool_kwargs,
                    tool_id=tool_id or tool_call.tool_id,
                    output=output,
                )
            )
    else:
        output = await acall_tool(tool, tool_call.tool_kwargs)

    if verbose:
        print("=== Function Output ===")
        print(output.content)

    return output
