from abc import abstractmethod
from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseComponent

DEFAULT_CHAT_STORE_KEY = "chat_history"


class BaseMemory(BaseComponent):
    """Base class for all memory types.

    NOTE: The interface for memory is not yet finalized and is subject to change.
    """

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "BaseMemory"

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
