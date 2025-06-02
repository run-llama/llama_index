import json
import uuid
from typing import Union, Dict, Any, List

from ag_ui.core import RunAgentInput
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
    CodeActAgent,
    AgentInput,
    AgentStream,
    ToolCall,
)
from llama_index.core.chat_engine.types import ChatMessage
from llama_index.core.workflow import Context, Event
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
    MessagesSnapshotWorkflowEvent,
    RunStartedWorkflowEvent,
    RunFinishedWorkflowEvent,
    RunErrorWorkflowEvent,
)
from llama_index.protocols.ag_ui.utils import (
    ag_ui_message_to_llama_index_message,
    get_kwargs_delta,
    llama_index_message_to_ag_ui_message,
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
    MessagesSnapshotWorkflowEvent,
    RunStartedWorkflowEvent,
    RunFinishedWorkflowEvent,
)


class AgentWorkflowRouter:
    def __init__(
        self, agent: Union[AgentWorkflow, FunctionAgent, ReActAgent, CodeActAgent]
    ):
        self.agent = agent
        self.router = APIRouter()
        self.router.add_api_route("/run", self.run, methods=["POST"])
        self.active_tool_calls = {}

    async def run(self, input: RunAgentInput):
        llama_index_messages = [
            ag_ui_message_to_llama_index_message(message) for message in input.messages
        ]
        state = input.state

        ctx = Context(self.agent)

        # set state if it exists
        if state:
            state_dict = json.loads(state)
            await ctx.set("state", state_dict)
        else:
            state_dict = {}

        chat_history = llama_index_messages[:-1]
        user_message = llama_index_messages[-1]
        last_message_id = input.messages[-1].id

        handler = self.agent.run(
            user_msg=user_message, chat_history=chat_history, ctx=ctx
        )

        async def stream_response():
            response_id = str(uuid.uuid4())
            current_state = state_dict.copy()

            try:
                yield workflow_event_to_sse(
                    RunStartedWorkflowEvent(
                        timestamp=timestamp(),
                        thread_id=input.thread_id,
                        run_id=input.run_id,
                    )
                )

                async for ev in handler.stream_events():
                    events_to_emit = []

                    # Handle state changes
                    events_to_emit.extend(
                        await self._handle_state_changes(ctx, current_state)
                    )

                    # Handle changes related to the event emitted
                    if isinstance(ev, AG_UI_EVENTS):
                        events_to_emit.append(ev)
                    elif isinstance(ev, AgentInput):
                        events_to_emit.extend(
                            self._handle_chat_history_changes(
                                ev.input, chat_history, last_message_id, response_id
                            )
                        )
                    elif isinstance(ev, AgentStream):
                        events_to_emit.extend(
                            self._handle_agent_stream(ev, response_id)
                        )
                    elif isinstance(ev, ToolCall):
                        events_to_emit.extend(self._handle_tool_call(ev, response_id))

                    # Emit all generated events
                    for event in events_to_emit:
                        yield workflow_event_to_sse(event)

                # Finish the run
                _ = await handler

                # Check for any remaining changes
                events_to_emit = await self._handle_state_changes(ctx, current_state)

                for event in events_to_emit:
                    yield workflow_event_to_sse(event)

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
                raise
            finally:
                self.active_tool_calls.pop(response_id, None)

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async def _handle_state_changes(
        self, ctx: Context, current_state: Dict[str, Any]
    ) -> list[Event]:
        """Handle state changes and emit state snapshot events."""
        state = await ctx.get("state", default={})
        changed_state = {}
        for key, value in state.items():
            if value != current_state.get(key):
                changed_state[key] = value

        current_state.clear()
        current_state.update(state)

        if changed_state:
            return [
                StateSnapshotWorkflowEvent(
                    timestamp=timestamp(),
                    snapshot=json.dumps(
                        changed_state, sort_keys=True, separators=(",", ":")
                    ),
                )
            ]
        return []

    def _handle_chat_history_changes(
        self,
        input: List[ChatMessage],
        chat_history: List[ChatMessage],
        last_message_id: str,
        output_message_id: str,
    ) -> list[Event]:
        """Handle chat history changes and emit messages snapshot events."""
        # Get the last message from chat history and input with the same id
        last_input_idx = next(
            (
                i
                for i, msg in enumerate(input)
                if msg.additional_kwargs.get("id") == last_message_id
            ),
            None,
        )
        if last_input_idx is None:
            return []

        # Check for new messages in the input
        new_messages = input[last_input_idx + 1 :]

        # If new messages are found, emit a messages snapshot event
        if new_messages:
            all_messages = chat_history + new_messages
            current_tool_calls = list(
                self.active_tool_calls.get(output_message_id, {}).values()
            )
            return [
                MessagesSnapshotWorkflowEvent(
                    timestamp=timestamp(),
                    messages=[
                        llama_index_message_to_ag_ui_message(m, current_tool_calls)
                        for m in all_messages
                    ],
                )
            ]

        return []

    def _handle_agent_stream(
        self, ev: AgentStream, output_message_id: str
    ) -> list[Event]:
        """Handle AgentStream events."""
        events = []

        # Handle text content
        if ev.delta:
            events.append(
                TextMessageChunkWorkflowEvent(
                    timestamp=timestamp(),
                    message_id=output_message_id,
                    delta=ev.delta,
                    role="assistant",
                )
            )

        # Handle tool calls
        for tool_call in ev.tool_calls:
            events.extend(self._handle_tool_call(tool_call, output_message_id))

        return events

    def _handle_tool_call(self, ev: ToolCall, parent_message_id: str) -> list[Event]:
        """Handle standalone ToolCall events (complete tool calls)."""
        tool_id = ev.tool_id
        old_tool_call = self.active_tool_calls.get(parent_message_id, {}).get(
            tool_id, {}
        )
        old_kwargs = old_tool_call.tool_kwargs if old_tool_call else {}

        self.active_tool_calls[parent_message_id][tool_id] = ev

        kwargs_delta = get_kwargs_delta(
            json.dumps(ev.tool_kwargs, sort_keys=True, separators=(",", ":")),
            json.dumps(old_kwargs, sort_keys=True, separators=(",", ":"))
            if old_kwargs
            else "",
        )

        return [
            ToolCallChunkWorkflowEvent(
                timestamp=timestamp(),
                tool_call_id=tool_id,
                tool_call_name=ev.tool_name,
                parent_message_id=parent_message_id,
                delta=kwargs_delta,
            )
        ]


def get_ag_ui_agent_router(
    agent: Union[AgentWorkflow, FunctionAgent, ReActAgent, CodeActAgent],
) -> APIRouter:
    server = AgentWorkflowRouter(agent)
    return server.router
