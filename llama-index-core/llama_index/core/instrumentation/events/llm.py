from llama_index.core.instrumentation.events.base import BaseEvent


class LLMPredictStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMPredictStartEvent"


class LLMPredictEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMPredictEndEvent"
