from abc import abstractmethod
from typing import List, Optional

from pydantic.v1 import BaseModel

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
        """Create a chat memory from defualts."""
        pass

    @abstractmethod
    def get(self) -> List[ChatMessage]:
        """Get chat history."""
        pass

    @abstractmethod
    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""
        pass

    @abstractmethod
    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        pass

    @abstractmethod
    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset chat history."""
        pass
