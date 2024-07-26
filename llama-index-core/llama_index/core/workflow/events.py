from collections import UserDict
from dataclasses import dataclass, field
from typing import Any


class Event:
    """Base class for event types."""


class StartEvent(UserDict, Event):
    """StartEvent is implicitly sent when a workflow runs."""


@dataclass
class StopEvent(Event):
    """EndEvent signals the workflow to stop."""

    result: Any = field(default=None)


EventType = type[Event]
