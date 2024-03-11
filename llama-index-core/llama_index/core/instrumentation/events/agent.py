from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.tools.types import ToolMetadata


class AgentRunStepStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "AgentRunStepStartEvent"


class AgentRunStepEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "AgentRunStepEndEvent"


class AgentChatWithStepStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "AgentChatWithStepStartEvent"


class AgentChatWithStepEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "AgentChatWithStepEndEvent"


class AgentToolCallEvent(BaseEvent):
    arguments: str
    tool: ToolMetadata

    @classmethod
    def class_name(cls):
        """Class name."""
        return "AgentToolCallEvent"
