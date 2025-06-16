from typing import Callable, Dict, Any, List, Optional, Awaitable

from ag_ui.core import RunAgentInput
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.workflow import Workflow
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow
from llama_index.protocols.ag_ui.events import (
    TextMessageStartWorkflowEvent,
    TextMessageContentWorkflowEvent,
    TextMessageChunkWorkflowEvent,
    TextMessageEndWorkflowEvent,
    ToolCallStartWorkflowEvent,
    ToolCallArgsWorkflowEvent,
    ToolCallChunkWorkflowEvent,
    ToolCallEndWorkflowEvent,
    StateSnapshotWorkflowEvent,
    StateDeltaWorkflowEvent,
    MessagesSnapshotWorkflowEvent,
    RunStartedWorkflowEvent,
    RunFinishedWorkflowEvent,
    RunErrorWorkflowEvent,
    CustomWorkflowEvent,
)
from llama_index.protocols.ag_ui.utils import (
    timestamp,
    workflow_event_to_sse,
)

AG_UI_EVENTS = (
    TextMessageStartWorkflowEvent,
    TextMessageContentWorkflowEvent,
    TextMessageEndWorkflowEvent,
    ToolCallStartWorkflowEvent,
    ToolCallArgsWorkflowEvent,
    ToolCallEndWorkflowEvent,
    StateSnapshotWorkflowEvent,
    StateDeltaWorkflowEvent,
    MessagesSnapshotWorkflowEvent,
    TextMessageChunkWorkflowEvent,
    ToolCallChunkWorkflowEvent,
    CustomWorkflowEvent,
)


class AGUIWorkflowRouter:
    def __init__(self, workflow_factory: Callable[[], Awaitable[Workflow]]):
        self.workflow_factory = workflow_factory
        self.router = APIRouter()
        self.router.add_api_route("/run", self.run, methods=["POST"])

    async def run(self, input: RunAgentInput):
        workflow = await self.workflow_factory()

        handler = workflow.run(
            input_data=input,
        )

        async def stream_response():
            try:
                yield workflow_event_to_sse(
                    RunStartedWorkflowEvent(
                        timestamp=timestamp(),
                        thread_id=input.thread_id,
                        run_id=input.run_id,
                    )
                )

                async for ev in handler.stream_events():
                    if isinstance(ev, AG_UI_EVENTS):
                        yield workflow_event_to_sse(ev)
                    else:
                        print(f"Unhandled event: {type(ev)}")

                # Finish the run
                _ = await handler

                # Update the state
                state = await handler.ctx.get("state", default={})
                yield workflow_event_to_sse(StateSnapshotWorkflowEvent(snapshot=state))

                yield workflow_event_to_sse(
                    RunFinishedWorkflowEvent(
                        timestamp=timestamp(),
                        thread_id=input.thread_id,
                        run_id=input.run_id,
                    )
                )
            except Exception as e:
                yield workflow_event_to_sse(
                    RunErrorWorkflowEvent(
                        timestamp=timestamp(),
                        message=str(e),
                        code=str(type(e)),
                    )
                )
                await handler.cancel_run()
                raise

        return StreamingResponse(stream_response(), media_type="text/event-stream")


def get_default_workflow_factory(
    llm: Optional[FunctionCallingLLM] = None,
    frontend_tools: Optional[List[str]] = None,
    backend_tools: Optional[List[str]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    timeout: Optional[float] = 120,
) -> Callable[[], Workflow]:
    async def workflow_factory():
        return AGUIChatWorkflow(
            llm=llm,
            frontend_tools=frontend_tools,
            backend_tools=backend_tools,
            initial_state=initial_state,
            system_prompt=system_prompt,
            timeout=timeout,
        )

    return workflow_factory


def get_ag_ui_workflow_router(
    workflow_factory: Optional[Callable[[], Awaitable[Workflow]]] = None,
    llm: Optional[FunctionCallingLLM] = None,
    frontend_tools: Optional[List[str]] = None,
    backend_tools: Optional[List[str]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    timeout: Optional[float] = 120,
) -> APIRouter:
    workflow_factory = workflow_factory or get_default_workflow_factory(
        llm, frontend_tools, backend_tools, initial_state, system_prompt, timeout
    )
    return AGUIWorkflowRouter(workflow_factory).router
