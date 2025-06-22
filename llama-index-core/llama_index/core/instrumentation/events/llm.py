from typing import Any, Dict, List, Optional
from llama_index.core.bridge.pydantic import BaseModel, SerializeAsAny, ConfigDict
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.prompts import BasePromptTemplate


class LLMPredictStartEvent(BaseEvent):
    """
    LLMPredictStartEvent.

    Args:
        template (BasePromptTemplate): Prompt template.
        template_args (Optional[dict]): Prompt template arguments.

    """

    template: SerializeAsAny[BasePromptTemplate]
    template_args: Optional[dict]

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMPredictStartEvent"


class LLMPredictEndEvent(BaseEvent):
    """
    LLMPredictEndEvent.

    The result of an llm.predict() call.

    Args:
        output (str): Output.

    """

    output: str

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMPredictEndEvent"


class LLMStructuredPredictStartEvent(BaseEvent):
    """
    LLMStructuredPredictStartEvent.

    Args:
        output_cls (Any): Output class to predict.
        template (BasePromptTemplate): Prompt template.
        template_args (Optional[dict]): Prompt template arguments.

    """

    output_cls: Any
    template: SerializeAsAny[BasePromptTemplate]
    template_args: Optional[dict]

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMStructuredPredictStartEvent"


class LLMStructuredPredictEndEvent(BaseEvent):
    """
    LLMStructuredPredictEndEvent.

    Args:
        output (BaseModel): Predicted output class.

    """

    output: SerializeAsAny[Any]

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMStructuredPredictEndEvent"


class LLMStructuredPredictInProgressEvent(BaseEvent):
    """
    LLMStructuredPredictInProgressEvent.

    Args:
        output (BaseModel): Predicted output class.

    """

    output: SerializeAsAny[Any]

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMStructuredPredictInProgressEvent"


class LLMCompletionStartEvent(BaseEvent):
    """
    LLMCompletionStartEvent.

    Args:
        prompt (str): The prompt to be completed.
        additional_kwargs (dict): Additional keyword arguments.
        model_dict (dict): Model dictionary.

    """

    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    prompt: str
    additional_kwargs: dict
    model_dict: dict

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMCompletionStartEvent"


class LLMCompletionInProgressEvent(BaseEvent):
    """
    LLMCompletionInProgressEvent.

    Args:
        prompt (str): The prompt to be completed.
        response (CompletionResponse): Completion response.

    """

    prompt: str
    response: CompletionResponse

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMCompletionInProgressEvent"

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        if isinstance(self.response.raw, BaseModel):
            self.response.raw = self.response.raw.model_dump()

        return super().model_dump(**kwargs)


class LLMCompletionEndEvent(BaseEvent):
    """
    LLMCompletionEndEvent.

    Args:
        prompt (str): The prompt to be completed.
        response (CompletionResponse): Completion response.

    """

    prompt: str
    response: CompletionResponse

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMCompletionEndEvent"

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        if isinstance(self.response.raw, BaseModel):
            self.response.raw = self.response.raw.model_dump()

        return super().model_dump(**kwargs)


class LLMChatStartEvent(BaseEvent):
    """
    LLMChatStartEvent.

    Args:
        messages (List[ChatMessage]): List of chat messages.
        additional_kwargs (dict): Additional keyword arguments.
        model_dict (dict): Model dictionary.

    """

    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    messages: List[ChatMessage]
    additional_kwargs: dict
    model_dict: dict

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMChatStartEvent"


class LLMChatInProgressEvent(BaseEvent):
    """
    LLMChatInProgressEvent.

    Args:
        messages (List[ChatMessage]): List of chat messages.
        response (ChatResponse): Chat response currently being streamed.

    """

    messages: List[ChatMessage]
    response: ChatResponse

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMChatInProgressEvent"

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        if isinstance(self.response.raw, BaseModel):
            self.response.raw = self.response.raw.model_dump()

        return super().model_dump(**kwargs)


class LLMChatEndEvent(BaseEvent):
    """
    LLMChatEndEvent.

    Args:
        messages (List[ChatMessage]): List of chat messages.
        response (Optional[ChatResponse]): Last chat response.

    """

    messages: List[ChatMessage]
    response: Optional[ChatResponse]

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLMChatEndEvent"

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        if self.response is not None and isinstance(self.response.raw, BaseModel):
            self.response.raw = self.response.raw.model_dump()

        return super().model_dump(**kwargs)
