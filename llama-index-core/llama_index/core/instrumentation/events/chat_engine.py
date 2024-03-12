from llama_index.core.instrumentation.events.base import BaseEvent


class StreamChatStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "StreamChatStartEvent"


class StreamChatEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "StreamChatEndEvent"


class StreamChatErrorEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "StreamChatErrorEvent"


class StreamChatDeltaReceivedEvent(BaseEvent):
    delta: str

    @classmethod
    def class_name(cls):
        """Class name."""
        return "StreamChatDeltaReceivedEvent"
