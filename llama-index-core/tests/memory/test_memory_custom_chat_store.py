"""Tests that Memory works with any AsyncDBChatStore implementation."""

from typing import Dict, List, Optional

import pytest

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field
from llama_index.core.memory.memory import Memory
from llama_index.core.storage.chat_store.base_db import (
    AsyncDBChatStore,
    MessageStatus,
)


class InMemoryChatStore(AsyncDBChatStore):
    """A minimal non-SQL AsyncDBChatStore, as a custom backend would implement it."""

    active: Dict[str, List[ChatMessage]] = Field(default_factory=dict)
    archived: Dict[str, List[ChatMessage]] = Field(default_factory=dict)

    def _bucket(self, key: str, status: MessageStatus) -> List[ChatMessage]:
        bucket = self.active if status == MessageStatus.ACTIVE else self.archived
        return bucket.setdefault(key, [])

    async def get_messages(
        self,
        key: str,
        status: Optional[MessageStatus] = MessageStatus.ACTIVE,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ChatMessage]:
        if status is None:
            messages = self._bucket(key, MessageStatus.ACTIVE) + self._bucket(
                key, MessageStatus.ARCHIVED
            )
        else:
            messages = list(self._bucket(key, status))

        messages = messages[offset or 0 :]
        if limit is not None:
            messages = messages[:limit]
        return messages

    async def count_messages(
        self, key: str, status: Optional[MessageStatus] = MessageStatus.ACTIVE
    ) -> int:
        return len(await self.get_messages(key, status=status))

    async def add_message(
        self,
        key: str,
        message: ChatMessage,
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        self._bucket(key, status).append(message)

    async def add_messages(
        self,
        key: str,
        messages: List[ChatMessage],
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        self._bucket(key, status).extend(messages)

    async def set_messages(
        self,
        key: str,
        messages: List[ChatMessage],
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        self._bucket(key, status)[:] = messages

    async def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        messages = self._bucket(key, MessageStatus.ACTIVE)
        if idx >= len(messages):
            return None
        return messages.pop(idx)

    async def delete_messages(
        self, key: str, status: Optional[MessageStatus] = None
    ) -> None:
        statuses = [status] if status is not None else list(MessageStatus)
        for message_status in statuses:
            self._bucket(key, message_status).clear()

    async def delete_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        messages = self._bucket(key, MessageStatus.ACTIVE)
        oldest = messages[:n]
        del messages[:n]
        return oldest

    async def archive_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        oldest = await self.delete_oldest_messages(key, n)
        self._bucket(key, MessageStatus.ARCHIVED).extend(oldest)
        return oldest

    async def get_keys(self) -> List[str]:
        return list({*self.active, *self.archived})


@pytest.fixture()
def chat_store():
    return InMemoryChatStore()


@pytest.mark.asyncio
async def test_memory_accepts_custom_chat_store(chat_store):
    """Memory should accept any AsyncDBChatStore, not just SQLAlchemyChatStore."""
    memory = Memory(token_limit=1000, session_id="test_user", sql_store=chat_store)

    await memory.aput_messages(
        [
            ChatMessage(role="user", content="Message 1"),
            ChatMessage(role="assistant", content="Response 1"),
        ]
    )

    messages = await memory.aget()
    assert [message.content for message in messages] == ["Message 1", "Response 1"]
    assert memory.sql_store is chat_store
    assert len(chat_store.active["test_user"]) == 2


@pytest.mark.asyncio
async def test_from_defaults_with_custom_chat_store(chat_store):
    """from_defaults should use the given chat store instead of building a SQL one."""
    memory = Memory.from_defaults(
        session_id="test_user",
        token_limit=1000,
        chat_history=[ChatMessage(role="user", content="Seeded")],
        chat_store=chat_store,
    )

    assert memory.sql_store is chat_store
    messages = await memory.aget()
    assert [message.content for message in messages] == ["Seeded"]


def test_from_defaults_without_chat_store_still_uses_sql_store():
    """The default backend is unchanged when no chat store is passed."""
    from llama_index.core.storage.chat_store.sql import SQLAlchemyChatStore

    memory = Memory.from_defaults(token_limit=1000, table_name="test_memory")

    assert isinstance(memory.sql_store, SQLAlchemyChatStore)
    assert memory.sql_store.table_name == "test_memory"


@pytest.mark.asyncio
async def test_waterfall_archives_into_custom_chat_store(chat_store):
    """Flushed messages are archived through the custom store."""
    memory = Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_user",
        sql_store=chat_store,
    )

    await memory.aput_messages(
        [
            ChatMessage(role="user", content="x " * 500),
            ChatMessage(role="assistant", content="y " * 500),
            ChatMessage(role="user", content="z " * 500),
        ]
    )

    messages = await memory.aget()
    assert len(messages) == 1
    assert "z " in messages[0].content
    assert len(chat_store.archived["test_user"]) == 2
