from abc import abstractmethod
from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseComponent
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.bridge.pydantic import Field, field_serializer, SerializeAsAny

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
        **kwargs: Any,
    ) -> "BaseMemory":
        """Create a chat memory from defaults."""

    @abstractmethod
    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""

    @abstractmethod
    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""

    @abstractmethod
    def put(self, message: ChatMessage) -> None:
        """Put chat history."""

    async def aput(self, message: ChatMessage) -> None:
        """Put chat history."""
        self.put(message)

    def put_messages(self, messages: List[ChatMessage]) -> None:
        """Put chat history."""
        for message in messages:
            self.put(message)

    @abstractmethod
    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""

    @abstractmethod
    def reset(self) -> None:
        """Reset chat history."""


class BaseChatStoreMemory(BaseMemory):
    """Base class for any .

    NOTE: The interface for memory is not yet finalized and is subject to change.
    """

    chat_store: SerializeAsAny[BaseChatStore] = Field(default_factory=SimpleChatStore)
    chat_store_key: str = Field(default=DEFAULT_CHAT_STORE_KEY)

    @field_serializer("chat_store")
    def serialize_courses_in_order(self, chat_store: BaseChatStore) -> dict:
        res = chat_store.model_dump()
        res.update({"class_name": chat_store.class_name()})
        return res

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "BaseChatStoreMemory"

    @classmethod
    @abstractmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "BaseChatStoreMemory":
        """Create a chat memory from defaults."""

    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""
        return self.chat_store.get_messages(self.chat_store_key)

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        # ensure everything is serialized
        self.chat_store.add_message(self.chat_store_key, message)

    async def aput(self, message: ChatMessage) -> None:
        """Put chat history."""
        # ensure everything is serialized
        await self.chat_store.async_add_message(self.chat_store_key, message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        self.chat_store.set_messages(self.chat_store_key, messages)

    def reset(self) -> None:
        """Reset chat history."""
        self.chat_store.delete_messages(self.chat_store_key)
