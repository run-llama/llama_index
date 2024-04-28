from llama_index.core.instrumentation.events.base import BaseEvent


class SpanDropEvent(BaseEvent):
    err_str: str

    @classmethod
    def class_name(cls):
        """Class name."""
        return "SpanDropEvent"
