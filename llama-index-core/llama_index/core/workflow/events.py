from collections import UserDict


class Event:
    """Base class for event types."""


class StartEvent(UserDict, Event):
    """StartEvent is implicitly sent when a workflow runs."""


class StopEvent(Event):
    """EndEvent signals the workflow to stop."""


EventType = type[Event]
