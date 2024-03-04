from llama_index.core.instrumentation.events.base import BaseEvent


class SynthesizeStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "SynthesizeStartEvent"


class SynthesizeEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "SynthesizeEndEvent"


class GetResponseStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "GetResponseStartEvent"


class GetResponseEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "GetResponseEndEvent"
