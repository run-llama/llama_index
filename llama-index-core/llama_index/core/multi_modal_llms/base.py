from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Union

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    ImageBlock,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_INPUT_FILES,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.schema import BaseComponent, ImageNode


class MultiModalLLMMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    context_window: Optional[int] = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input when generating a response."
        ),
    )
    num_output: Optional[int] = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    num_input_files: Optional[int] = Field(
        default=DEFAULT_NUM_INPUT_FILES,
        description="Number of input files the model can take when generating a response.",
    )
    is_function_calling_model: Optional[bool] = Field(
        default=False,
        # SEE: https://openai.com/blog/function-calling-and-other-api-updates
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )

    is_chat_model: bool = Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )


class MultiModalLLM(BaseComponent, DispatcherSpanMixin):
    """Multi-Modal LLM interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Help static checkers understand this class hierarchy
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi-Modal LLM metadata."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """Streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat endpoint for Multi-Modal LLM."""

    # ===== Async Endpoints =====

    @abstractmethod
    async def acomplete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for Multi-Modal LLM."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        The callback decorators installs events, so they must be applied before
        the span decorators, otherwise the spans wouldn't contain the events.
        """
        for attr in (
            "complete",
            "acomplete",
            "stream_complete",
            "astream_complete",
            "chat",
            "achat",
            "stream_chat",
            "astream_chat",
        ):
            if callable(method := cls.__dict__.get(attr)):
                if attr.endswith("chat"):
                    setattr(cls, attr, llm_chat_callback()(method))
                else:
                    setattr(cls, attr, llm_completion_callback()(method))
        super().__init_subclass__(**kwargs)
