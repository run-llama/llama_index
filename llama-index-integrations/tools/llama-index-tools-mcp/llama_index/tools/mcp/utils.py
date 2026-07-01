from typing import Any, Callable, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow
from llama_index.tools.mcp.base import McpToolSpec
from llama_index.tools.mcp.client import BasicMCPClient


def get_tools_from_mcp_url(
    command_or_url: str,
    client: Optional[ClientSession] = None,
    allowed_tools: Optional[List[str]] = None,
    global_partial_params: Optional[Dict[str, Any]] = None,
    partial_params_by_tool: Optional[Dict[str, Dict[str, Any]]] = None,
    include_resources: bool = False,
) -> List[FunctionTool]:
    """
    Get tools from an MCP server or command.

    Args:
        command_or_url: The command to run or the URL to connect to.
        client (optional): The client to use to connect to the MCP server.
        allowed_tools (optional): The tool names to allow from the MCP server.
        global_partial_params: A dict of params to apply to all tools globally.
        partial_params_by_tool: A dict mapping tool names to param overrides.
                                Values override global_partial_params. Use None as a value to remove a global param for a specific tool.
        include_resources (optional): Whether to include resources in the tool list.

    """
    client = client or BasicMCPClient(command_or_url)
    tool_spec = McpToolSpec(
        client,
        allowed_tools=allowed_tools,
        global_partial_params=global_partial_params,
        partial_params_by_tool=partial_params_by_tool,
        include_resources=include_resources,
    )
    return tool_spec.to_tool_list()


async def aget_tools_from_mcp_url(
    command_or_url: str,
    client: Optional[ClientSession] = None,
    allowed_tools: Optional[List[str]] = None,
    global_partial_params: Optional[Dict[str, Any]] = None,
    partial_params_by_tool: Optional[Dict[str, Dict[str, Any]]] = None,
    include_resources: bool = False,
) -> List[FunctionTool]:
    """
    Get tools from an MCP server or command.

    Args:
        command_or_url: The command to run or the URL to connect to.
        client (optional): The client to use to connect to the MCP server.
        allowed_tools (optional): The tool names to allow from the MCP server.
        global_partial_params: A dict of params to apply to all tools globally.
        partial_params_by_tool: A dict mapping tool names to param overrides.
                                Values override global_partial_params. Use None as a value to remove a global param for a specific tool.
        include_resources (optional): Whether to include resources in the tool list.

    """
    client = client or BasicMCPClient(command_or_url)
    tool_spec = McpToolSpec(
        client,
        allowed_tools=allowed_tools,
        global_partial_params=global_partial_params,
        partial_params_by_tool=partial_params_by_tool,
        include_resources=include_resources,
    )
    return await tool_spec.to_tool_list_async()


def workflow_as_mcp(
    workflow: Optional[Workflow] = None,
    workflow_name: Optional[str] = None,
    workflow_description: Optional[str] = None,
    start_event_model: Optional[BaseModel] = None,
    workflow_factory: Optional[Callable[[], Workflow]] = None,
    **fastmcp_init_kwargs: Any,
) -> FastMCP:
    """
    Convert a workflow to an MCP app.

    This will convert any `Workflow` to an MCP app. It will expose the workflow as a tool
    within MCP, which will

    Args:
        workflow (optional):
            The workflow instance to convert. This instance is reused for every MCP tool call.
        workflow_name (optional):
            The name of the workflow. Defaults to the workflow class name.
        workflow_description (optional):
            The description of the workflow. Defaults to the workflow docstring.
        start_event_model (optional):
            The start event model of the workflow. Can be a `BaseModel` or a `StartEvent` class.
            Defaults to the workflow's custom `StartEvent` class.
        workflow_factory (optional):
            Factory that creates a fresh workflow instance for each MCP tool call.
        **fastmcp_init_kwargs:
            Additional keyword arguments to pass to the FastMCP constructor.

    Returns:
        The MCP app object.

    """
    if workflow is None and workflow_factory is None:
        raise ValueError("Must provide either workflow or workflow_factory.")
    if workflow is not None and workflow_factory is not None:
        raise ValueError("Provide either workflow or workflow_factory, not both.")

    app = FastMCP(**fastmcp_init_kwargs)

    # Dynamically get the start event class -- this is a bit of a hack
    if start_event_model is None:
        if workflow is None:
            raise ValueError(
                "Must provide start_event_model when using workflow_factory without a workflow instance."
            )
        StartEventCLS = workflow._start_event_class
    else:
        StartEventCLS = start_event_model

    if StartEventCLS == StartEvent:
        raise ValueError(
            "Must declare a custom StartEvent class in your workflow or provide a start_event_model."
        )

    # Get the workflow name and description
    if workflow_name is None:
        workflow_name = (
            workflow.__class__.__name__
            if workflow is not None
            else getattr(workflow_factory, "__name__", None)
        )
    if workflow_name is None:
        raise ValueError("Must provide workflow_name when it cannot be inferred.")

    workflow_description = (
        workflow_description
        if workflow_description is not None
        else workflow.__doc__
        if workflow is not None
        else None
    )

    @app.tool(name=workflow_name, description=workflow_description)
    async def _workflow_tool(run_args: StartEventCLS, context: Context) -> Any:
        # Handle edge cases where the start event is an Event or a BaseModel
        # If the workflow does not have a custom StartEvent class, then we need to handle the event differently
        active_workflow = (
            workflow_factory() if workflow_factory is not None else workflow
        )
        if active_workflow is None:
            raise ValueError("Must provide either workflow or workflow_factory.")

        if (
            isinstance(run_args, Event)
            and active_workflow._start_event_class != StartEvent
        ):
            handler = active_workflow.run(start_event=run_args)
        elif isinstance(run_args, BaseModel):
            handler = active_workflow.run(**run_args.model_dump())
        elif isinstance(run_args, dict):
            start_event = StartEventCLS.model_validate(run_args)
            handler = active_workflow.run(start_event=start_event)
        else:
            raise ValueError(f"Invalid start event type: {type(run_args)}")

        async for event in handler.stream_events():
            if not isinstance(event, StopEvent):
                await context.log("info", message=event.model_dump_json())

        return await handler

    return app
