from llama_index.core.instrumentation.events.base import BaseEvent


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
