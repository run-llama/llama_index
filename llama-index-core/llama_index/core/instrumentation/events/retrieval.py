from llama_index.core.instrumentation.events.base import BaseEvent


class RetrievalStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "RetrievalStartEvent"


class RetrievalEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "RetrievalEndEvent"
