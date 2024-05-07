from llama_index.core.instrumentation.events.base import BaseEvent


class StreamChatStartEvent(BaseEvent):
    """StreamChatStartEvent.

    Fired at the start of writing to the stream chat-engine queue.
    """

    @classmethod
    def class_name(cls):
        """Class name."""
        return "StreamChatStartEvent"


class StreamChatEndEvent(BaseEvent):
    """StreamChatEndEvent.

    Fired at the end of writing to the stream chat-engine queue.
    """

    @classmethod
    def class_name(cls):
        """Class name."""
        return "StreamChatEndEvent"


class StreamChatErrorEvent(BaseEvent):
    """StreamChatErrorEvent.

    Fired when an exception is raised during the stream chat-engine operation.

    Args:
        exception (Exception): Exception raised during the stream chat operation.
    """

    exception: Exception

    @classmethod
    def class_name(cls):
        """Class name."""
        return "StreamChatErrorEvent"


class StreamChatDeltaReceivedEvent(BaseEvent):
    """StreamChatDeltaReceivedEvent.

    Args:
        delta (str): Delta received from the stream chat.
    """

    delta: str

    @classmethod
    def class_name(cls):
        """Class name."""
        return "StreamChatDeltaReceivedEvent"
