from abc import abstractmethod
from typing import Any, Optional, Sequence

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_INPUT_FILES,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.schema import BaseComponent, ImageDocument


class MultiModalLLMMetadata(BaseModel):
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


# TODO add callback functionality


class MultiModalLLM(BaseComponent):
    """Multi-Modal LLM interface."""

    class Config:
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi-Modal LLM metadata."""

    @abstractmethod
    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
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
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
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
