import json
import uuid
from typing import Union, Dict, Any, List

from ag_ui.core import RunAgentInput, FunctionCall, AssistantMessage
from ag_ui.core.types import ToolCall as AgUIToolCall
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
    CodeActAgent,
    AgentStream,
    ToolCall,
    ToolCallResult,
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


class MessageIDFactory:
    """Used to generate IDs for messages while a workflow is running."""

    def __init__(self, initial_ids: List[str] = None):
        self.current_ids = initial_ids or [str(uuid.uuid4())]

    def next_id(self) -> str:
        new_id = str(uuid.uuid4())
        self.current_ids.append(new_id)
        return new_id

    def cur_id(self) -> str:
        return self.current_ids[-1]


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
        if state and isinstance(state, str):
            state_dict = json.loads(state)
            await ctx.set("state", state_dict)
        elif state and isinstance(state, dict):
            state.pop("messages", None)
            state_dict = state.copy()
            await ctx.set("state", state)
        else:
            state_dict = {}

        chat_history = llama_index_messages[:-1]
        user_message = llama_index_messages[-1]
        last_message_id = input.messages[-1].id

        handler = self.agent.run(
            user_msg=user_message, chat_history=[*chat_history], ctx=ctx
        )

        async def stream_response():
            id_factory = MessageIDFactory(initial_ids=[m.id for m in input.messages])
            # create current ID for the agent response (needed so that tool calls have a parent)
            id_factory.next_id()

            current_state = state_dict.copy()
            latest_input: List[ChatMessage] = []
            prev_ev_type = None

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
                    # elif isinstance(ev, AgentInput):
                    #    latest_input = ev.input
                    #    events_to_emit.extend(
                    #        self._handle_chat_history_changes(
                    #            ev.input, chat_history, last_message_id, id_factory
                    #        )
                    #    )
                    elif isinstance(ev, AgentStream):
                        # Increment the ID if we switched from a tool call to an agent stream
                        if prev_ev_type is type(ToolCall):
                            id_factory.next_id()

                        events_to_emit.extend(
                            self._handle_agent_stream(ev, id_factory.cur_id())
                        )
                    elif isinstance(ev, ToolCall) and not isinstance(
                        ev, ToolCallResult
                    ):
                        events_to_emit.extend(
                            self._handle_tool_call(ev, id_factory.cur_id())
                        )

                    # Track previous event type
                    prev_ev_type = type(ev)

                    # Emit all generated events
                    for event in events_to_emit:
                        yield workflow_event_to_sse(event)

                # Finish the run
                _ = await handler

                # Check for any remaining changes
                events_to_emit = await self._handle_state_changes(ctx, current_state)

                # Handle final chat history update
                # events_to_emit.extend(
                #    self._handle_chat_history_changes(
                #        [*latest_input, response.response],
                #        chat_history,
                #        last_message_id,
                #        id_factory,
                #    )
                # )

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
                for response_id in id_factory.current_ids:
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
        id_factory: MessageIDFactory,
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
        new_messages = input[last_input_idx:]

        # If new messages are found, emit a messages snapshot event
        if new_messages and len(new_messages) > 1:
            all_messages = [*chat_history, *new_messages]

            # Update message IDs if needed
            num_ids_to_add = len(all_messages) - len(id_factory.current_ids)
            if num_ids_to_add > 0:
                for _ in range(num_ids_to_add):
                    id_factory.next_id()

            # Align messages with the current IDs
            for msg, id in zip(all_messages, id_factory.current_ids):
                msg.additional_kwargs["id"] = id

            # Pass current tool calls to the last message only
            message_snapshot = MessagesSnapshotWorkflowEvent(
                timestamp=timestamp(),
                messages=[
                    llama_index_message_to_ag_ui_message(m) for m in all_messages
                ],
            )

            # Insert current tool calls into the last assistant message
            for msg in message_snapshot.messages:
                if msg.id in self.active_tool_calls and isinstance(
                    msg, AssistantMessage
                ):
                    msg.tool_calls = [
                        AgUIToolCall(
                            type="function",
                            id=tool_call.tool_id,
                            function=FunctionCall(
                                name=tool_call.tool_name,
                                arguments=json.dumps(
                                    tool_call.tool_kwargs, sort_keys=True
                                ),
                            ),
                        )
                        for tool_call in self.active_tool_calls.get(msg.id, {}).values()
                    ]

            return []

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

        return events

    def _handle_tool_call(self, ev: ToolCall, parent_message_id: str) -> list[Event]:
        """Handle standalone ToolCall events (complete tool calls)."""
        tool_id = ev.tool_id
        tool_kwargs = ev.tool_kwargs

        if parent_message_id not in self.active_tool_calls:
            self.active_tool_calls[parent_message_id] = {}

        self.active_tool_calls[parent_message_id][tool_id] = ev

        return [
            ToolCallChunkWorkflowEvent(
                timestamp=timestamp(),
                tool_call_id=tool_id,
                tool_call_name=ev.tool_name,
                parent_message_id=parent_message_id,
                delta=json.dumps(tool_kwargs, sort_keys=True, separators=(",", ":")),
            )
        ]


def get_ag_ui_agent_router(
    agent: Union[AgentWorkflow, FunctionAgent, ReActAgent, CodeActAgent],
) -> APIRouter:
    server = AgentWorkflowRouter(agent)
    return server.router
