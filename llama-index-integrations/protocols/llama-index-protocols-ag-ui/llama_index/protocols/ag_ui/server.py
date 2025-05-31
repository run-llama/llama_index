import json
import uuid
from typing import Union, Optional, Dict, Any, List
from enum import Enum

from ag_ui.core import RunAgentInput
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
    CodeActAgent,
    AgentStream,
    ToolCall,
)
from llama_index.core.workflow import Context, Event
from llama_index.protocols.ag_ui.events import (
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


class EventState(Enum):
    IDLE = "idle"
    TEXT_MESSAGE_ACTIVE = "text_message_active"
    TOOL_CALL_ACTIVE = "tool_call_active"


class EventStateManager:
    """Manages the state transitions for AG-UI events to ensure only one event type is active at a time."""

    def __init__(self):
        self.state = EventState.IDLE
        self.current_message_id: Optional[str] = None
        self.current_tool_call_id: Optional[str] = None
        self.parent_message_id: Optional[str] = None
        self.active_tool_calls: Dict[str, ToolCall] = {}
        self.processed_tool_calls: set[str] = set()
        self.tool_call_last_sent: Dict[str, str] = {}
        self.tool_call_seen_values: Dict[str, set] = {}

    def get_pending_events_to_end(self) -> list[Event]:
        """Get events needed to cleanly end any active state and clear the state."""
        events = []

        if self.state == EventState.TEXT_MESSAGE_ACTIVE and self.current_message_id:
            events.append(
                TextMessageEndWorkflowEvent(
                    timestamp=timestamp(),
                    message_id=self.current_message_id,
                )
            )
            # Clear the state after getting the event
            self.parent_message_id = self.current_message_id
            self.current_message_id = None
        elif self.state == EventState.TOOL_CALL_ACTIVE and self.current_tool_call_id:
            events.append(
                ToolCallEndWorkflowEvent(
                    timestamp=timestamp(),
                    tool_call_id=self.current_tool_call_id,
                )
            )
            # DON'T clear active_tool_calls here - only clear current state
            self.current_tool_call_id = None

        self.state = EventState.IDLE
        return events

    def start_text_message(self) -> TextMessageStartWorkflowEvent:
        """Start a new text message. Note: caller should end any active events first."""
        self.current_message_id = str(uuid.uuid4())
        self.state = EventState.TEXT_MESSAGE_ACTIVE

        return TextMessageStartWorkflowEvent(
            timestamp=timestamp(),
            message_id=self.current_message_id,
            role="assistant",
        )

    def add_text_content(self, delta: str) -> Optional[TextMessageContentWorkflowEvent]:
        """Add content to the current text message."""
        if self.state != EventState.TEXT_MESSAGE_ACTIVE or not self.current_message_id:
            return None

        return TextMessageContentWorkflowEvent(
            timestamp=timestamp(),
            message_id=self.current_message_id,
            delta=delta,
        )

    def start_tool_call(
        self, tool_call: ToolCall
    ) -> Optional[ToolCallStartWorkflowEvent]:
        """Start a new tool call. Note: caller should end any active events first."""
        if tool_call.tool_id in self.processed_tool_calls:
            return None

        self.current_tool_call_id = tool_call.tool_id
        self.active_tool_calls[tool_call.tool_id] = tool_call
        self.tool_call_last_sent[tool_call.tool_id] = ""
        self.tool_call_seen_values[tool_call.tool_id] = set()
        self.state = EventState.TOOL_CALL_ACTIVE

        return ToolCallStartWorkflowEvent(
            timestamp=timestamp(),
            tool_call_id=tool_call.tool_id,
            tool_call_name=tool_call.tool_name,
            parent_message_id=self.parent_message_id or str(uuid.uuid4()),
        )

    def update_tool_call_args(
        self, tool_call: ToolCall
    ) -> Optional[ToolCallArgsWorkflowEvent]:
        """Update tool call arguments with incremental JSON fragments."""
        if tool_call.tool_id not in self.active_tool_calls:
            return None

        import json

        # Track which fields we've seen with non-null values
        seen_values = self.tool_call_seen_values.get(tool_call.tool_id, set())

        # Update our tracking of which fields have had real values
        for key, value in tool_call.tool_kwargs.items():
            if value is not None:
                seen_values.add(key)

        # Smart filtering: include nulls only for fields we've already seen with real values
        # This way we don't send "placeholder" nulls, but we do send "final" nulls
        smart_kwargs = {}
        for key, value in tool_call.tool_kwargs.items():
            if value is not None or key in seen_values:
                smart_kwargs[key] = value

        # Only proceed if we have data to send
        if not smart_kwargs:
            return None

        current_json = json.dumps(
            smart_kwargs, separators=(",", ":"), ensure_ascii=False
        )
        last_sent = self.tool_call_last_sent.get(tool_call.tool_id, "")

        # Calculate delta using the existing utility function
        from llama_index.protocols.ag_ui.utils import get_kwargs_delta

        delta = get_kwargs_delta(current_json, last_sent)

        if delta:
            # Update what we've sent and our value tracking
            self.tool_call_last_sent[tool_call.tool_id] = current_json
            self.tool_call_seen_values[tool_call.tool_id] = seen_values

            # Update stored tool call
            self.active_tool_calls[tool_call.tool_id] = tool_call

            return ToolCallArgsWorkflowEvent(
                timestamp=timestamp(),
                tool_call_id=tool_call.tool_id,
                delta=delta,
            )
        return None

    def end_tool_call(self, tool_call_id: str) -> Optional[List[Event]]:
        """End a specific tool call."""
        if tool_call_id not in self.active_tool_calls:
            return None
        if tool_call_id in self.processed_tool_calls:
            return None

        self.active_tool_calls.pop(tool_call_id, None)
        self.tool_call_last_sent.pop(tool_call_id, None)
        self.tool_call_seen_values.pop(tool_call_id, None)  # Clean up tracking
        self.processed_tool_calls.add(tool_call_id)

        if self.current_tool_call_id == tool_call_id:
            self.current_tool_call_id = None
            self.state = EventState.IDLE

        return [
            ToolCallArgsWorkflowEvent(
                timestamp=timestamp(),
                tool_call_id=tool_call_id,
                delta="}",
            ),
            ToolCallEndWorkflowEvent(
                timestamp=timestamp(),
                tool_call_id=tool_call_id,
            ),
        ]


class AgentWorkflowRouter:
    def __init__(
        self, agent: Union[AgentWorkflow, FunctionAgent, ReActAgent, CodeActAgent]
    ):
        self.agent = agent
        self.router = APIRouter()
        self.router.add_api_route("/run", self.run, methods=["POST"])

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

        handler = self.agent.run(
            user_msg=user_message, chat_history=chat_history, ctx=ctx
        )

        async def stream_response():
            current_state = state_dict.copy()
            event_manager = EventStateManager()

            try:
                yield workflow_event_to_sse(
                    RunStartedWorkflowEvent(
                        timestamp=timestamp(),
                        thread_id=input.thread_id,
                        run_id=input.run_id,
                    )
                )

                async for ev in handler.stream_events():
                    # Handle state changes
                    events_to_emit = await self._handle_state_changes(
                        ctx, current_state
                    )

                    # Handle chat history changes
                    events_to_emit.extend(
                        await self._handle_chat_history_changes(ctx, chat_history)
                    )

                    # Process the current event
                    events_to_emit.extend(await self._process_event(ev, event_manager))

                    # Emit all generated events
                    for event in events_to_emit:
                        yield workflow_event_to_sse(event)

                # Ensure all events are properly closed
                for event in event_manager.get_pending_events_to_end():
                    yield workflow_event_to_sse(event)

                # Finish the run
                _ = await handler

                # Check for any remaining changes
                events_to_emit = await self._handle_state_changes(ctx, current_state)
                events_to_emit.extend(
                    await self._handle_chat_history_changes(ctx, chat_history)
                )

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

        events = []
        if changed_state:
            events.append(
                StateSnapshotWorkflowEvent(
                    timestamp=timestamp(),
                    snapshot=json.dumps(
                        changed_state, sort_keys=True, separators=(",", ":")
                    ),
                )
            )
        return events

    async def _handle_chat_history_changes(
        self, ctx: Context, chat_history: list
    ) -> list[Event]:
        """Handle chat history changes and emit messages snapshot events."""
        events = []
        memory = await ctx.get("memory", default=None)
        if memory:
            current_chat_history = await memory.aget_all()
            if len(current_chat_history) != len(chat_history):
                new_messages = [
                    llama_index_message_to_ag_ui_message(message)
                    for message in current_chat_history
                ]
                events.append(
                    MessagesSnapshotWorkflowEvent(
                        timestamp=timestamp(),
                        messages=new_messages,
                    )
                )
        return events

    async def _process_event(
        self, ev: Event, event_manager: EventStateManager
    ) -> list[Event]:
        """Process a single event and return the events to emit."""
        events_to_emit = []

        # Handle AG-UI events directly
        if isinstance(ev, AG_UI_EVENTS):
            events_to_emit.append(ev)

        # Handle AgentStream events
        elif isinstance(ev, AgentStream):
            events_to_emit.extend(self._handle_agent_stream(ev, event_manager))

        # Handle ToolCall events (complete tool calls)
        elif isinstance(ev, ToolCall):
            events_to_emit.extend(self._handle_tool_call(ev, event_manager))

        return events_to_emit

    def _handle_agent_stream(
        self, ev: AgentStream, event_manager: EventStateManager
    ) -> list[Event]:
        """Handle AgentStream events."""
        events = []

        # Handle text content
        if ev.delta:
            if event_manager.state != EventState.TEXT_MESSAGE_ACTIVE:
                events.extend(event_manager.get_pending_events_to_end())
                events.append(event_manager.start_text_message())

            content_event = event_manager.add_text_content(ev.delta)
            if content_event:
                events.append(content_event)

        # Handle tool calls
        for tool_call in ev.tool_calls:
            # ensure we have created at least one message before starting a tool call
            if (
                event_manager.current_message_id is None
                and event_manager.parent_message_id is None
            ):
                events.append(event_manager.start_text_message())
                events.extend(event_manager.get_pending_events_to_end())

            if tool_call.tool_id not in event_manager.active_tool_calls:
                # End any active events before starting NEW tool call
                events.extend(event_manager.get_pending_events_to_end())

                # Start new tool call
                start_event = event_manager.start_tool_call(tool_call)
                if start_event:
                    events.append(start_event)

                    # Add initial args
                    if tool_call.tool_kwargs:
                        events.append(
                            ToolCallArgsWorkflowEvent(
                                timestamp=timestamp(),
                                tool_call_id=tool_call.tool_id,
                                delta=json.dumps(
                                    tool_call.tool_kwargs,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        )
            else:
                # Tool call already exists - only update if we're currently in tool call state
                if (
                    event_manager.state == EventState.TOOL_CALL_ACTIVE
                    and event_manager.current_tool_call_id == tool_call.tool_id
                ):
                    args_event = event_manager.update_tool_call_args(tool_call)
                    if args_event:
                        events.append(args_event)
                # If the tool call exists but isn't the current active one, ignore it
                # (it's probably a duplicate from another event stream)

        return events

    def _handle_tool_call(
        self, ev: ToolCall, event_manager: EventStateManager
    ) -> list[Event]:
        """Handle standalone ToolCall events (complete tool calls)."""
        events = []

        # Check if we were already streaming this tool call
        if ev.tool_id in event_manager.active_tool_calls:
            # End the tool call
            end_events = event_manager.end_tool_call(ev.tool_id)
            if end_events:
                events.extend(end_events)
        else:
            # New tool call we haven't seen before - start, send args, and end

            # End any active events before starting tool call
            events.extend(event_manager.get_pending_events_to_end())

            # Start the tool call
            start_event = event_manager.start_tool_call(ev)
            if start_event:
                events.append(start_event)

                # Add the tool call arguments
                if ev.tool_kwargs:
                    events.append(
                        ToolCallArgsWorkflowEvent(
                            timestamp=timestamp(),
                            tool_call_id=ev.tool_id,
                            delta=json.dumps(
                                ev.tool_kwargs, sort_keys=True, separators=(",", ":")
                            ),
                        )
                    )

                # End the tool call immediately since it's complete
                end_events = event_manager.end_tool_call(ev.tool_id)
                if end_events:
                    events.extend(end_events)

        return events


def get_ag_ui_agent_router(
    agent: Union[AgentWorkflow, FunctionAgent, ReActAgent, CodeActAgent],
) -> APIRouter:
    server = AgentWorkflowRouter(agent)
    return server.router
