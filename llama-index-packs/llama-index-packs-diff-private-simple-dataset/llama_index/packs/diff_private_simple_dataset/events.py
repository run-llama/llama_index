from typing import List
from llama_index.core.instrumentation.events import BaseEvent


class LLMEmptyResponseEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMEmptyResponseEvent"


class EmptyIntersectionEvent(BaseEvent):
    public_tokens: List[str]
    private_tokens: List[str]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "EmptyIntersectionEvent"


class SyntheticExampleStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "SyntheticExampleStartEvent"


class SyntheticExampleEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "SyntheticExampleEndEvent"
