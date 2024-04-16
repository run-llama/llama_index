from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryType


class SynthesizeStartEvent(BaseEvent):
    query: QueryType

    @classmethod
    def class_name(cls):
        """Class name."""
        return "SynthesizeStartEvent"


class SynthesizeEndEvent(BaseEvent):
    query: QueryType
    response: RESPONSE_TYPE

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
