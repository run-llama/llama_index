from typing import List

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
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


class LLMStructuredPredictStartEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMStructuredPredictStartEvent"


class LLMStructuredPredictEndEvent(BaseEvent):
    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMStructuredPredictEndEvent"


class LLMCompletionStartEvent(BaseEvent):
    prompt: str
    additional_kwargs: dict
    model_dict: dict

    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMCompletionStartEvent"


class LLMCompletionEndEvent(BaseEvent):
    prompt: str
    response: CompletionResponse

    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMCompletionEndEvent"


class LLMChatStartEvent(BaseEvent):
    messages: List[ChatMessage]
    additional_kwargs: dict
    model_dict: dict

    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMChatStartEvent"


class LLMChatInProgressEvent(BaseEvent):
    messages: List[ChatMessage]
    response: ChatResponse

    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMChatInProgressEvent"


class LLMChatEndEvent(BaseEvent):
    messages: List[ChatMessage]
    response: ChatResponse

    @classmethod
    def class_name(cls):
        """Class name."""
        return "LLMChatEndEvent"
