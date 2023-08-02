from abc import abstractmethod
from pydantic import BaseModel
from typing import List, Optional

from llama_index.llms.base import ChatMessage, LLM


class BaseMemory(BaseModel):
    """Base class for all memory types.

    NOTE: The interface for memory is not yet finalized and is subject to change.
    """

    def to_string(self) -> str:
        """Convert memory to string."""
        return self.json()

    @classmethod
    def from_string(cls, json_str: str) -> "BaseMemory":
        return cls.parse_raw(json_str)

    def to_dict(self) -> dict:
        """Convert memory to dict."""
        return self.dict()

    @classmethod
    def from_dict(cls, json_dict: dict) -> "BaseMemory":
        return cls.parse_obj(json_dict)

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
