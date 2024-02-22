"""Events for FunctionCall."""

from dataclasses import dataclass
from typing import Any, Dict

from deprecated import deprecated

from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType
from llama_index.core.tools.types import ToolMetadata


@dataclass
class FunctionCallStartEventPayload:
    """Payload for FunctionCallStartEvent."""

    function_call: Dict[str, Any]
    tool: ToolMetadata


class FunctionCallStartEvent(CBEvent):
    """Event to indicate a function call has started."""

    function_call: Dict[str, Any]
    tool: ToolMetadata

    def __init__(self, function_call: Dict[str, Any], tool: ToolMetadata):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.RERANKING)
        self.function_call = function_call
        self.tool = tool

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> FunctionCallStartEventPayload:
        """Return the payload for the event."""
        return FunctionCallStartEventPayload(
            function_call=self.function_call, tool=self.tool
        )


@dataclass
class FunctionCallEndEventPayload:
    """Payload for FunctionCallEndEvent."""

    function_output: str


class FunctionCallEndEvent(CBEvent):
    """Event to indicate a function call has ended."""

    function_output: str

    def __init__(self, function_output: str):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.RERANKING)
        self.function_output = function_output

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> FunctionCallEndEventPayload:
        """Return the payload for the event."""
        return FunctionCallEndEventPayload(function_output=self.function_output)
