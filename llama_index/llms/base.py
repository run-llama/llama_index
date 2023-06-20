from abc import ABC
from typing import Generator, Optional, Sequence, Union

from pydantic import BaseModel, Field


# ===== Generic Model Input - Chat =====
class Message(BaseModel):
    content: str
    additional_kwargs: dict = Field(default_factory=dict)


class ChatMessage(Message):
    role: str


class FunctionMessage(ChatMessage):
    name: str


# ===== Generic Model Output - Chat =====
class ChatResponse(BaseModel):
    message: Message
    raw: Optional[dict] = None


class ChatDeltaResponse(ChatResponse):
    delta: str


# ===== Generic Model Output - Completion =====
class CompletionResponse(BaseModel):
    text: str
    raw: Optional[dict] = None


class CompletionDeltaResponse(CompletionResponse):
    delta: str


CompletionResponseType = Union[
    CompletionResponse, Generator[CompletionDeltaResponse, None, None]
]
ChatResponseType = Union[ChatResponse, Generator[ChatDeltaResponse, None, None]]


class LLM(ABC):
    def chat(messages: Sequence[Message]) -> ChatResponseType:
        pass

    def complete(prompt: str) -> CompletionResponseType:
        pass

    # ===== Async Endpoints =====
    async def achat(messages: Sequence[Message]) -> ChatResponseType:
        pass

    async def acomplete(prompt: str) -> CompletionResponseType:
        pass
