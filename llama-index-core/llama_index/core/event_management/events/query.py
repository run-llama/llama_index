from llama_index.core.event_management.events.base import BaseEvent


class QueryStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "QueryStartEvent"


class QueryEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "QueryEndEvent"
