from llama_index.core.instrumentation.events.base import BaseEvent


class SynthesizeStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "SynthesizeStartEvent"


class RetrievalEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "RetrievalEndEvent"
