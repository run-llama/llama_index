"""Abstract Base Event."""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from deprecated import deprecated

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.events.base_event_type import CBEventType

# timestamp for callback events
TIMESTAMP_FORMAT = "%m/%d/%Y, %H:%M:%S.%f"


class CBEvent(BaseModel, ABC):
    """Base class to store event information."""

    _event_type: CBEventType
    _time: str = ""
    _id: str = ""

    def __init__(self, event_type: CBEventType) -> None:
        """Init time and id if needed."""
        self._event_type = event_type
        self._time = datetime.now().strftime(TIMESTAMP_FORMAT)
        self._id = str(uuid.uuid4())

    @property
    def id_(self) -> str:
        """Return the UUID id for the event."""
        return self._id

    @property
    def time(self) -> str:
        """Return the timestamp for the event."""
        return self._time

    @property
    @deprecated("You can call isinstance on the class to get the type of class/event.")
    def event_type(self) -> CBEventType:
        """Return the event type."""
        return self._event_type

    @property
    @abstractmethod
    @deprecated("You can access the payload properties directly from the class.")
    def payload(self) -> Any:
        """Return the payload for the event (to support legacy systems)."""
