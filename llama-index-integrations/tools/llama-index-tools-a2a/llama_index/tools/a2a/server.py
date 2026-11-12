"""
Inbound A2A Server - exposes a LlamaIndex Workflow as an A2A agent.

Mirrors the `workflow_as_mcp` pattern from llama-index-tools-mcp.
Uses the a2a-sdk server components (AgentExecutor, DefaultRequestHandler, etc.)
to create a FastAPI app that conforms to the A2A protocol.
"""

from typing import Any, Dict, Optional, List, Awaitable, Callable

from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    AgentProvider,
    Part,
    TextPart,
    Message,
    Role,
    Task,
    TaskStatus,
    TaskState,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    Artifact,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AFastAPIApplication
from pydantic import BaseModel

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow


class WorkflowAgentExecutor(AgentExecutor):
    """
    AgentExecutor that wraps a LlamaIndex Workflow.

    The workflow's StartEvent fields become the input schema,
    and the workflow execution produces the output streamed via events.
    """

    def __init__(
        self,
        workflow: Workflow,
        start_event_model: Optional[type[BaseModel]] = None,
        workflow_name: Optional[str] = None,
        workflow_description: Optional[str] = None,
    ):
        self.workflow = workflow
        self.start_event_model = start_event_model or workflow._start_event_class
        self.workflow_name = workflow_name or workflow.__class__.__name__
        self.workflow_description = workflow_description or workflow.__doc__ or "LlamaIndex Workflow"

        if self.start_event_model == StartEvent:
            raise ValueError(
                "Must declare a custom StartEvent class in your workflow "
                "or provide a start_event_model."
            )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the workflow, streaming intermediate and final events."""
        user_input = context.get_user_input()
        if not user_input:
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.failed),
                final=True,
            ))
            return

        try:
            # Parse input into the workflow's StartEvent model
            import json
            if isinstance(user_input, str):
                try:
                    input_data = json.loads(user_input)
                except json.JSONDecodeError:
                    # Single-argument workflow
                    fields = self.start_event_model.model_fields
                    if len(fields) == 1:
                        field_name = list(fields.keys())[0]
                        input_data = {field_name: user_input}
                    else:
                        raise ValueError(f"Expected JSON input, got: {user_input}")
            elif isinstance(user_input, dict):
                input_data = user_input
            else:
                input_data = {"input": user_input}

            # Validate and create StartEvent
            start_event = self.start_event_model.model_validate(input_data)

            # Run the workflow (Workflow.run() returns a coroutine — await it directly)
            result = await self.workflow.run(start_event=start_event)

            # Send artifact with result
            if result is not None:
                try:
                    result_text = result.model_dump_json()
                except AttributeError:
                    result_text = str(result)

                await event_queue.enqueue_event(TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    artifact=Artifact(
                        artifact_id=f"{context.task_id}-artifact",
                        parts=[TextPart(text=result_text)],
                    ),
                    last_chunk=True,
                ))

            # Final completion status
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.completed),
                final=True,
            ))

        except Exception as e:
            await event_queue.enqueue_event(TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=Message(
                        messageId=f"{context.task_id}-error",
                        role=Role.agent,
                        parts=[TextPart(text=f"Workflow error: {str(e)}")],
                    ),
                ),
                final=True,
            ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the workflow execution."""
        await event_queue.enqueue_event(TaskStatusUpdateEvent(
            task_id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.canceled),
            final=True,
        ))


def create_agent_card(
    workflow: Workflow,
    workflow_name: Optional[str] = None,
    workflow_description: Optional[str] = None,
    start_event_model: Optional[type[BaseModel]] = None,
    url: str = "http://localhost:8000",
    version: str = "1.0.0",
) -> AgentCard:
    """Create an AgentCard from a LlamaIndex Workflow."""
    name = workflow_name or workflow.__class__.__name__
    description = workflow_description or workflow.__doc__ or "LlamaIndex Workflow Agent"

    StartEventCLS = start_event_model or workflow._start_event_class

    if StartEventCLS != StartEvent:
        input_schema = StartEventCLS.model_json_schema()
    else:
        input_schema = {"type": "object", "properties": {}}

    skill = AgentSkill(
        id=name.lower().replace(" ", "-"),
        name=name,
        description=description,
        tags=["workflow", "llama-index"],
        examples=[f"Run {name} workflow"],
        input_modes=["text", "application/json"],
        output_modes=["text", "application/json"],
    )

    capabilities = AgentCapabilities(streaming=True, push_notifications=False)
    provider = AgentProvider(organization="LlamaIndex", url="https://github.com/run-llama/llama_index")

    agent_card = AgentCard(
        name=name,
        description=description,
        version=version,
        url=url,
        skills=[skill],
        capabilities=capabilities,
        provider=provider,
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
    )

    return agent_card


def workflow_as_a2a(
    workflow: Workflow,
    workflow_name: Optional[str] = None,
    workflow_description: Optional[str] = None,
    start_event_model: Optional[type[BaseModel]] = None,
    url: str = "http://localhost:8000",
    version: str = "1.0.0",
    **app_kwargs: Any,
) -> Any:
    """
    Convert a LlamaIndex Workflow to an A2A-compatible FastAPI application.

    Mirrors the `workflow_as_mcp` pattern from llama-index-tools-mcp.

    Args:
        workflow: The LlamaIndex Workflow to expose as an A2A agent.
        workflow_name: Name for the agent (defaults to workflow class name).
        workflow_description: Description (defaults to workflow docstring).
        start_event_model: The StartEvent model for the workflow.
                           Defaults to workflow._start_event_class.
        url: Base URL where the agent will be hosted.
        version: Version string for the agent card.
        **app_kwargs: Additional kwargs passed to FastAPI constructor.

    Returns:
        A FastAPI application with A2A routes configured.

    Example:
        >>> from llama_index.core.workflow import Workflow, StartEvent, StopEvent
        >>> from llama_index.tools.a2a import workflow_as_a2a
        >>>
        >>> class MyStartEvent(StartEvent):
        ...     query: str
        >>>
        >>> class MyWorkflow(Workflow):
        ...     async def run(self, start_event: MyStartEvent) -> StopEvent:
        ...         return StopEvent(result=f"Processed: {start_event.query}")
        >>>
        >>> app = workflow_as_a2a(MyWorkflow())
        >>> # Run: uvicorn my_app:app --host 0.0.0.0 --port 8000
    """
    # Create AgentCard
    agent_card = create_agent_card(
        workflow=workflow,
        workflow_name=workflow_name,
        workflow_description=workflow_description,
        start_event_model=start_event_model,
        url=url,
        version=version,
    )

    # Create executor
    executor = WorkflowAgentExecutor(
        workflow=workflow,
        start_event_model=start_event_model,
        workflow_name=workflow_name,
        workflow_description=workflow_description,
    )

    # Create task store
    task_store = InMemoryTaskStore()

    # Create request handler (agent_card handled by A2AFastAPIApplication separately)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # Build the A2A FastAPI application
    app = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build(**app_kwargs)

    # Store references on app state for programmatic access
    app.state.agent_card = agent_card
    app.state.executor = executor
    app.state.task_store = task_store
    app.state.request_handler = request_handler

    return app
