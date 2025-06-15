# Converts ag_ui events to llama_index workflow events

from ag_ui.core.events import (
    EventType,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageChunkEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    StateSnapshotEvent,
    StateDeltaEvent,
    MessagesSnapshotEvent,
    RawEvent,
    CustomEvent,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
)

from llama_index.core.workflow import Event


class TextMessageStartWorkflowEvent(TextMessageStartEvent, Event):
    type: EventType = EventType.TEXT_MESSAGE_START


class TextMessageContentWorkflowEvent(TextMessageContentEvent, Event):
    type: EventType = EventType.TEXT_MESSAGE_CONTENT


class TextMessageChunkWorkflowEvent(TextMessageChunkEvent, Event):
    type: EventType = EventType.TEXT_MESSAGE_CHUNK


class TextMessageEndWorkflowEvent(TextMessageEndEvent, Event):
    type: EventType = EventType.TEXT_MESSAGE_END


class ToolCallStartWorkflowEvent(ToolCallStartEvent, Event):
    type: EventType = EventType.TOOL_CALL_START


class ToolCallArgsWorkflowEvent(ToolCallArgsEvent, Event):
    type: EventType = EventType.TOOL_CALL_ARGS


class ToolCallChunkWorkflowEvent(ToolCallChunkEvent, Event):
    type: EventType = EventType.TOOL_CALL_CHUNK


class ToolCallEndWorkflowEvent(ToolCallEndEvent, Event):
    type: EventType = EventType.TOOL_CALL_END


class StateSnapshotWorkflowEvent(StateSnapshotEvent, Event):
    type: EventType = EventType.STATE_SNAPSHOT


class StateDeltaWorkflowEvent(StateDeltaEvent, Event):
    type: EventType = EventType.STATE_DELTA


class MessagesSnapshotWorkflowEvent(MessagesSnapshotEvent, Event):
    type: EventType = EventType.MESSAGES_SNAPSHOT


class RawWorkflowEvent(RawEvent, Event):
    type: EventType = EventType.RAW


class CustomWorkflowEvent(CustomEvent, Event):
    type: EventType = EventType.CUSTOM


class RunStartedWorkflowEvent(RunStartedEvent, Event):
    type: EventType = EventType.RUN_STARTED


class RunFinishedWorkflowEvent(RunFinishedEvent, Event):
    type: EventType = EventType.RUN_FINISHED


class RunErrorWorkflowEvent(RunErrorEvent, Event):
    type: EventType = EventType.RUN_ERROR


class StepStartedWorkflowEvent(StepStartedEvent, Event):
    type: EventType = EventType.STEP_STARTED


class StepFinishedWorkflowEvent(StepFinishedEvent, Event):
    type: EventType = EventType.STEP_FINISHED
