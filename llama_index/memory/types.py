from abc import abstractmethod
from typing import Any, List, Optional

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.base import LLM, ChatMessage


class BaseMemory(BaseModel):
    """Base class for all memory types.

    NOTE: The interface for memory is not yet finalized and is subject to change.
    """

    @classmethod
    @abstractmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        llm: Optional[LLM] = None,
    ) -> "BaseMemory":
        """Create a chat memory from defaults."""

    @abstractmethod
    def get(self, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""

    @abstractmethod
    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""

    @abstractmethod
    def put(self, message: ChatMessage) -> None:
        """Put chat history."""

    @abstractmethod
    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""

    @abstractmethod
    def reset(self) -> None:
        """Reset chat history."""
