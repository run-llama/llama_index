"""Tests that Memory works with any AsyncDBChatStore implementation."""

from typing import Dict, List, Optional

import asyncio

import pytest

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, PrivateAttr
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


class StaggeredArchiveChatStore(InMemoryChatStore):
    """
    Store that staggers concurrent archive calls.

    While armed, the first archive call yields briefly before reading the
    active queue, and any later archive call yields for longer. Without
    serialization in Memory this forces the classic interleaving: both
    _manage_queue calls snapshot the queue before either archives, then the
    second archive runs against rows the first one already flushed, so its
    positional "oldest n" lands on messages that were meant to stay active.
    """

    _armed: bool = PrivateAttr(default=False)
    _archive_calls: int = PrivateAttr(default=0)

    async def archive_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        if self._armed:
            self._archive_calls += 1
            await asyncio.sleep(0.05 if self._archive_calls == 1 else 0.3)
        return await super().archive_oldest_messages(key, n)


def _seed_messages() -> List[ChatMessage]:
    # ~12 tokens each with the default tokenizer
    return [
        ChatMessage(role="user", content=f"seed{i} " + "word " * 12) for i in range(6)
    ]


@pytest.mark.asyncio
async def test_concurrent_aput_does_not_over_archive() -> None:
    """
    Two concurrent puts on one session must archive the same messages as
    back-to-back puts. Previously each waterfall snapshotted the queue before
    either archived, and the second positional archive flushed messages that
    were supposed to stay active.
    """
    session_id = "race_user"

    sequential_store = InMemoryChatStore()
    await sequential_store.add_messages(session_id, _seed_messages())
    sequential_memory = Memory(
        token_limit=60,
        token_flush_size=40,
        chat_history_token_ratio=1.0,
        session_id=session_id,
        sql_store=sequential_store,
    )
    await sequential_memory.aput(
        ChatMessage(role="user", content="newA " + "word " * 12)
    )
    await sequential_memory.aput(
        ChatMessage(role="user", content="newB " + "word " * 12)
    )
    expected_active = [
        m.content for m in await sequential_store.get_messages(session_id)
    ]
    # the sequential run really does flush some seeds, so the assertion below
    # is sensitive to what the waterfall does
    assert len(expected_active) < 8

    gated_store = StaggeredArchiveChatStore()
    await gated_store.add_messages(session_id, _seed_messages())
    gated_store._armed = True
    concurrent_memory = Memory(
        token_limit=60,
        token_flush_size=40,
        chat_history_token_ratio=1.0,
        session_id=session_id,
        sql_store=gated_store,
    )
    await asyncio.gather(
        concurrent_memory.aput(
            ChatMessage(role="user", content="newA " + "word " * 12)
        ),
        concurrent_memory.aput(
            ChatMessage(role="user", content="newB " + "word " * 12)
        ),
    )
    gated_store._armed = False

    actual_active = [m.content for m in await gated_store.get_messages(session_id)]
    assert actual_active == expected_active


def test_sync_put_works_across_event_loops() -> None:
    """
    The sync wrappers spin up a fresh event loop per call; the manage lock
    must follow the running loop instead of binding to the first one.
    """
    memory = Memory(
        token_limit=1000, session_id="loop_user", sql_store=InMemoryChatStore()
    )

    memory.put(ChatMessage(role="user", content="one"))
    memory.put(ChatMessage(role="user", content="two"))

    assert [m.content for m in memory.get_all()] == ["one", "two"]
