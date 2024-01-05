"""Base interface class for storing chat history per user."""
from abc import abstractmethod
from typing import List, Optional

from llama_index.llms import ChatMessage
from llama_index.schema import BaseComponent


class BaseChatStore(BaseComponent):
    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "BaseChatStore"

    @abstractmethod
    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        ...

    @abstractmethod
    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        ...

    @abstractmethod
    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        ...

    @abstractmethod
    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete messages for a key."""
        ...

    @abstractmethod
    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        ...

    @abstractmethod
    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        ...

    @abstractmethod
    def get_keys(self) -> List[str]:
        """Get all keys."""
        ...
