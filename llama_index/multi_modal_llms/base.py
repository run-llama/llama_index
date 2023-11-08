from abc import abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Generator, Optional, Sequence

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_INPUT_FILES,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.schema import BaseComponent, ImageDocument


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


# ===== Generic Model Input - Chat =====
class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole = MessageRole.USER
    content: Optional[Any] = ""
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"


# ===== Generic Model Output - Completion =====
class MultiModalCompletionResponse(BaseModel):
    """
    Completion response.

    Fields:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        additional_kwargs: Additional information on the response(i.e. token
            counts, function calling information).
        raw: Optional raw JSON that was parsed to populate text, if relevant.
        delta: New text that just streamed in (only relevant when streaming).
    """

    text: str
    additional_kwargs: dict = Field(default_factory=dict)
    raw: Optional[dict] = None
    delta: Optional[str] = None

    def __str__(self) -> str:
        return self.text


MultiModalCompletionResponseGen = Generator[MultiModalCompletionResponse, None, None]
MultiModalCompletionResponseAsyncGen = AsyncGenerator[
    MultiModalCompletionResponse, None
]


class MultiModalLLMMetadata(BaseModel):
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input when generating a response."
        ),
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    num_input_files: int = Field(
        default=DEFAULT_NUM_INPUT_FILES,
        description="Number of input files the model can take when generating a response.",
    )
    is_function_calling_model: bool = Field(
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
    ) -> MultiModalCompletionResponse:
        """Completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> MultiModalCompletionResponseGen:
        """Streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> MultiModalCompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> MultiModalCompletionResponseAsyncGen:
        """Async streaming completion endpoint for Multi-Modal LLM."""
