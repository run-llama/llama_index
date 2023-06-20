from abc import ABC, abstractmethod
from typing import Generator, Optional, Sequence, Union

from pydantic import BaseModel, Field

from llama_index.llm_predictor.base import LLMMetadata


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


class LLM(ABC):

    @property
    @abstractmethod
    def metadata() -> LLMMetadata:
        pass

    @abstractmethod
    def chat(messages: Sequence[Message]) -> ChatResponseType:
        pass

    @abstractmethod
    def complete(prompt: str) -> CompletionResponseType:
        pass

    # ===== Async Endpoints =====
    @abstractmethod
    async def achat(messages: Sequence[Message]) -> ChatResponseType:
        pass

    @abstractmethod
    async def acomplete(prompt: str) -> CompletionResponseType:
        pass
