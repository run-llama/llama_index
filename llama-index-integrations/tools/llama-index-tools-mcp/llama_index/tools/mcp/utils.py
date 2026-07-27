from typing import Any, Callable, Dict, List, Optional, Type, Union
import warnings

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
    workflow: Union[Type[Workflow], Workflow, Callable[[], Workflow]],
    workflow_name: Optional[str] = None,
    workflow_description: Optional[str] = None,
    start_event_model: Optional[BaseModel] = None,
    share_instance: bool = False,
    **fastmcp_init_kwargs: Any,
) -> FastMCP:
    """
    Convert a workflow to an MCP app.

    This will convert any `Workflow` to an MCP app. It will expose the workflow as a tool
    within MCP, which will create a fresh workflow instance for each invocation to ensure
    complete isolation between different MCP clients.

    Args:
        workflow:
            The workflow to convert. Can be:
            - A Workflow class (e.g., CountingWorkflow)
            - A callable factory that returns a Workflow (e.g., lambda: CountingWorkflow(timeout=30))
            - A Workflow instance (will extract class; init params will be lost unless share_instance=True)
        workflow_name (optional):
            The name of the workflow. Defaults to the workflow class name.
        workflow_description (optional):
            The description of the workflow. Defaults to the workflow docstring.
        start_event_model (optional):
            The start event model of the workflow. Can be a `BaseModel` or a `StartEvent` class.
            Defaults to the workflow's custom `StartEvent` class.
        share_instance (optional):
            If True, reuses the same workflow instance across all MCP invocations.
            WARNING: This can cause state pollution and data leakage between different
            MCP clients/tenants. Only use this if you fully understand the implications.
            Defaults to False (secure by default).
        **fastmcp_init_kwargs:
            Additional keyword arguments to pass to the FastMCP constructor.

    Returns:
        The MCP app object.

    """
    app = FastMCP(**fastmcp_init_kwargs)

    # Determine the workflow factory and metadata instance
    if share_instance:
        # Opt-out: User explicitly wants to share state (old buggy behavior)
        if not isinstance(workflow, Workflow):
            raise ValueError(
                "share_instance=True requires a Workflow instance. "
                "If you want to share state, pass an instance like: workflow_as_mcp(MyWorkflow(), share_instance=True)"
            )
        workflow_factory = None  # Not used in share mode
        workflow_instance = workflow
    else:
        # SECURE BY DEFAULT: Create fresh instances per invocation
        if isinstance(workflow, type):
            # It's a class - use it as factory
            workflow_factory = workflow
            workflow_instance = workflow()  # For metadata extraction
        elif isinstance(workflow, Workflow):
            # It's an instance - extract class and warn about lost params
            # CHECK THIS BEFORE callable() to avoid treating instances as factories
            warnings.warn(
                f"Passing a Workflow instance to workflow_as_mcp will extract its class "
                f"and create fresh instances without initialization parameters. "
                f"To preserve init params, pass a lambda: workflow_as_mcp(lambda: {workflow.__class__.__name__}(...)). "
                f"To share state across invocations (NOT RECOMMENDED), use share_instance=True.",
                DeprecationWarning,
                stacklevel=2,
            )
            workflow_class = workflow.__class__
            workflow_factory = workflow_class
            workflow_instance = workflow
        elif callable(workflow):
            # It's a callable/lambda factory
            workflow_factory = workflow
            workflow_instance = workflow()  # Call once for metadata
        else:
            raise TypeError(
                f"workflow must be a Workflow class, instance, or callable factory. "
                f"Got {type(workflow).__name__}"
            )

    # Dynamically get the start event class -- this is a bit of a hack
    StartEventCLS = start_event_model or workflow_instance._start_event_class
    if StartEventCLS == StartEvent:
        raise ValueError(
            "Must declare a custom StartEvent class in your workflow or provide a start_event_model."
        )

    # Get the workflow name and description
    workflow_name = workflow_name or workflow_instance.__class__.__name__
    workflow_description = workflow_description or workflow_instance.__doc__

    @app.tool(name=workflow_name, description=workflow_description)
    async def _workflow_tool(run_args: StartEventCLS, context: Context) -> Any:
        # Determine which workflow instance to use
        if share_instance:
            # Use the shared instance (opt-out behavior - NOT RECOMMENDED)
            wf = workflow_instance
        else:
            # SECURE BY DEFAULT: Create a fresh workflow instance for each invocation
            # This ensures complete isolation between different MCP clients/invocations
            wf = workflow_factory()

        # Handle edge cases where the start event is an Event or a BaseModel
        # If the workflow does not have a custom StartEvent class, then we need to handle the event differently

        if isinstance(run_args, Event) and wf._start_event_class != StartEvent:
            handler = wf.run(start_event=run_args)
        elif isinstance(run_args, BaseModel):
            handler = wf.run(**run_args.model_dump())
        elif isinstance(run_args, dict):
            start_event = StartEventCLS.model_validate(run_args)
            handler = wf.run(start_event=start_event)
        else:
            raise ValueError(f"Invalid start event type: {type(run_args)}")

        async for event in handler.stream_events():
            if not isinstance(event, StopEvent):
                await context.log("info", message=event.model_dump_json())

        return await handler

    return app
