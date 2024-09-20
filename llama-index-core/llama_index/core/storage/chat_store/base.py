"""Base interface class for storing chat history per user."""
from abc import abstractmethod
from typing import List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.schema import BaseComponent


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

    async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Async version of Get messages for a key."""
        self.set_messages(key, messages)

    async def aget_messages(self, key: str) -> List[ChatMessage]:
        """Async version of Get messages for a key."""
        return self.get_messages(key)

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """Async version of Add a message for a key."""
        self.add_message(key, message)

    async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Async version of Delete messages for a key."""
        return self.delete_messages(key)

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Async version of Delete specific message for a key."""
        return self.delete_message(key, idx)

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Async version of Delete last message for a key."""
        return self.delete_last_message(key)

    async def aget_keys(self) -> List[str]:
        """Async version of Get all keys."""
        return self.get_keys()
