"""Events for Exceptions."""

from dataclasses import dataclass

from deprecated import deprecated

from llama_index.core.events.base_event import CBEvent
from llama_index.core.events.base_event_type import CBEventType


@dataclass
class ExceptionStartEventPayload:
    """Payload for ExceptionStartEvent."""

    exception: Exception


class ExceptionStartEvent(CBEvent):
    """Event to indicate an exception has happened."""

    exception: Exception

    def __init__(self, exception: Exception):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.EXCEPTION)
        self.exception = exception

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> ExceptionStartEventPayload:
        """Return the payload for the event."""
        return ExceptionStartEventPayload(exception=self.exception)


@dataclass
class ExceptionEndEventPayload:
    """Payload for ExceptionEndEvent."""

    exception: Exception


@deprecated("Exceptions are instant and have no end.")
class ExceptionEndEvent(CBEvent):
    """Event to indicate an exception has happened."""

    exception: Exception

    def __init__(self, exception: Exception):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.EXCEPTION)
        self.exception = exception

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> ExceptionStartEventPayload:
        """Return the payload for the event."""
        return ExceptionStartEventPayload(exception=self.exception)
