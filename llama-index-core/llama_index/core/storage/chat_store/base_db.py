from abc import abstractmethod
from typing import List, Optional
from enum import Enum

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import BaseModel


class MessageStatus(str, Enum):
    """Status of a message in the chat store."""

    # Message is in the active FIFO queue
    ACTIVE = "active"

    # Message has been processed and is archived, removed from the active queue
    ARCHIVED = "archived"


class AsyncDBChatStore(BaseModel):
    """
    Base class for DB-based chat stores.

    Meant to implement a FIFO queue to manage short-term memory and
    general conversation history.
    """

    @abstractmethod
    async def get_messages(
        self,
        key: str,
        status: Optional[MessageStatus] = MessageStatus.ACTIVE,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get all messages for a key with the specified status (async).

        Returns a list of messages.
        """

    @abstractmethod
    async def count_messages(
        self,
        key: str,
        status: Optional[MessageStatus] = MessageStatus.ACTIVE,
    ) -> int:
        """Count messages for a key with the specified status (async)."""

    @abstractmethod
    async def add_message(
        self,
        key: str,
        message: ChatMessage,
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Add a message for a key with the specified status (async)."""

    @abstractmethod
    async def add_messages(
        self,
        key: str,
        messages: List[ChatMessage],
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Add a list of messages in batch for the specified key and status (async)."""

    @abstractmethod
    async def set_messages(
        self,
        key: str,
        messages: List[ChatMessage],
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Set all messages for a key (replacing existing ones) with the specified status (async)."""

    @abstractmethod
    async def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a specific message by ID and return it (async)."""

    @abstractmethod
    async def delete_messages(
        self, key: str, status: Optional[MessageStatus] = None
    ) -> None:
        """Delete all messages for a key with the specified status (async)."""

    @abstractmethod
    async def delete_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        """Delete the oldest n messages for a key and return them (async)."""

    @abstractmethod
    async def archive_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        """Archive the oldest n messages for a key and return them (async)."""

    @abstractmethod
    async def get_keys(self) -> List[str]:
        """Get all unique keys in the store (async)."""

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "AsyncDBChatStore"
