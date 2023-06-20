from abc import ABC, abstractmethod
from typing import Any, Awaitable, Generator, Optional, Sequence, Union

from pydantic import BaseModel, Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS


# ===== Generic Model Input - Chat =====
class Message(BaseModel):
    content: str
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


class ChatMessage(Message):
    role: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class FunctionMessage(ChatMessage):
    name: str


# ===== Generic Model Output - Chat =====
class ChatResponse(BaseModel):
    message: Message
    raw: Optional[dict] = None

    def __str__(self) -> str:
        return str(self.message)


class ChatDeltaResponse(ChatResponse):
    delta: str

    def __str__(self) -> str:
        return self.delta


# ===== Generic Model Output - Completion =====
class CompletionResponse(BaseModel):
    text: str
    raw: Optional[dict] = None

    def __str__(self) -> str:
        return self.text


class CompletionDeltaResponse(CompletionResponse):
    delta: str

    def __str__(self) -> str:
        return self.delta


CompletionResponseType = Union[
    CompletionResponse, Generator[CompletionDeltaResponse, None, None]
]
ChatResponseType = Union[ChatResponse, Generator[ChatDeltaResponse, None, None]]


class LLMMetadata(BaseModel):
    """LLM metadata.

    We extract this metadata to help with our prompts.

    """

    context_window: int = DEFAULT_CONTEXT_WINDOW
    num_output: int = DEFAULT_NUM_OUTPUTS


class LLM(ABC):
    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        pass

    @abstractmethod
    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseType:
        pass

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponseType:
        pass

    # ===== Async Endpoints =====
    @abstractmethod
    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> Awaitable[ChatResponseType]:
        pass

    @abstractmethod
    async def acomplete(self, prompt: str, **kwargs) -> Awaitable[CompletionResponseType]:
        pass
