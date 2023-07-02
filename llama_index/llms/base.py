from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generator, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


# ===== Generic Model Input - Chat =====
class ChatMessage(BaseModel):
    role: MessageRole = MessageRole.USER
    content: Optional[str] = ""
    additional_kwargs: dict = Field(default_factory=dict)
    name: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"


# ===== Generic Model Output - Chat =====
class ChatResponse(BaseModel):
    message: ChatMessage
    raw: Optional[dict] = None
    delta: Optional[str] = None

    def __str__(self) -> str:
        return str(self.message)


ChatResponseGen = Generator[ChatResponse, None, None]

# ===== Generic Model Output - Completion =====
class CompletionResponse(BaseModel):
    text: str
    additional_kwargs: dict = Field(default_factory=dict)
    raw: Optional[dict] = None
    delta: Optional[str] = None

    def __str__(self) -> str:
        return self.text


CompletionResponseGen = Generator[CompletionResponse, None, None]


class LLMMetadata(BaseModel):
    """LLM metadata."""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    num_output: int = DEFAULT_NUM_OUTPUTS


class LLM(ABC):
    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        pass

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        pass

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        pass

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        pass

    @abstractmethod
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        pass

    # ===== Async Endpoints =====
    @abstractmethod
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        pass

    @abstractmethod
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        pass

    @abstractmethod
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        pass

    @abstractmethod
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        pass
