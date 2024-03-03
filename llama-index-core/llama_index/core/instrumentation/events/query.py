from llama_index.core.instrumentation.events.base import BaseEvent


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
