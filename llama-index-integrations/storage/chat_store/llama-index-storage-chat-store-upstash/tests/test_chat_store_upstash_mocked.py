"""Tests that run without a real Upstash instance, using in-memory fake clients."""

from typing import Dict, List, Optional

import pytest
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.upstash import UpstashChatStore


class FakeRedis:
    """Minimal in-memory stand-in for the list commands used by the store."""

    def __init__(self) -> None:
        self.data: Dict[str, List[str]] = {}

    def rpush(self, key: str, *values: str) -> int:
        self.data.setdefault(key, []).extend(values)
        return len(self.data[key])

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        items = self.data.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]

    def lindex(self, key: str, idx: int) -> Optional[str]:
        try:
            return self.data.get(key, [])[idx]
        except IndexError:
            return None

    def lset(self, key: str, idx: int, value: str) -> bool:
        self.data[key][idx] = value
        return True

    def lrem(self, key: str, count: int, value: str) -> int:
        items = self.data.get(key, [])
        if value in items:
            items.remove(value)
            return 1
        return 0

    def rpop(self, key: str) -> Optional[str]:
        items = self.data.get(key, [])
        return items.pop() if items else None

    def delete(self, *keys: str) -> int:
        return sum(1 for k in keys if self.data.pop(k, None) is not None)

    def expire(self, key: str, seconds: int) -> bool:
        return True

    def keys(self, pattern: str) -> List[str]:
        return list(self.data)


class FakeAsyncRedis:
    """Async wrapper around FakeRedis that records which commands were awaited."""

    def __init__(self, sync: FakeRedis) -> None:
        self._sync = sync
        self.calls: List[str] = []

    def __getattr__(self, name: str):
        method = getattr(self._sync, name)

        async def _call(*args, **kwargs):
            self.calls.append(name)
            return method(*args, **kwargs)

        return _call


@pytest.fixture()
def store() -> UpstashChatStore:
    chat_store = UpstashChatStore(redis_url="https://fake.upstash.io", redis_token="t")
    fake = FakeRedis()
    chat_store._sync_redis_client = fake
    chat_store._async_redis_client = FakeAsyncRedis(fake)
    return chat_store


def _msgs(*contents: str) -> List[ChatMessage]:
    return [ChatMessage(role="user", content=c) for c in contents]


def test_delete_message_returns_chat_message(store: UpstashChatStore) -> None:
    store.set_messages("k", _msgs("a", "b", "c"))

    deleted = store.delete_message("k", 1)

    assert isinstance(deleted, ChatMessage)
    assert deleted.content == "b"
    assert [m.content for m in store.get_messages("k")] == ["a", "c"]


def test_delete_message_missing_index_returns_none(store: UpstashChatStore) -> None:
    store.set_messages("k", _msgs("a"))
    assert store.delete_message("k", 5) is None


@pytest.mark.asyncio
async def test_adelete_message_returns_chat_message(store: UpstashChatStore) -> None:
    await store.aset_messages("k", _msgs("a", "b", "c"))

    deleted = await store.adelete_message("k", 1)

    assert isinstance(deleted, ChatMessage)
    assert deleted.content == "b"
    assert [m.content for m in await store.aget_messages("k")] == ["a", "c"]


@pytest.mark.parametrize(
    "method",
    [
        "aset_messages",
        "aget_messages",
        "adelete_messages",
        "adelete_message",
        "adelete_last_message",
        "aget_keys",
    ],
)
def test_overrides_base_async_methods(method: str) -> None:
    # The base class async API must be overridden, otherwise callers such as
    # chat memory fall back to running the sync client in a thread.
    assert getattr(UpstashChatStore, method) is not getattr(BaseChatStore, method)


@pytest.mark.asyncio
async def test_async_api_uses_async_client(store: UpstashChatStore) -> None:
    await store.aset_messages("k", _msgs("a", "b", "c"))
    assert [m.content for m in await store.aget_messages("k")] == ["a", "b", "c"]

    last = await store.adelete_last_message("k")
    assert last is not None and last.content == "c"

    assert await store.aget_keys() == ["k"]

    await store.adelete_messages("k")
    assert await store.aget_messages("k") == []

    calls = store._async_redis_client.calls
    for command in ("delete", "rpush", "lrange", "rpop", "keys"):
        assert command in calls


@pytest.mark.asyncio
async def test_async_add_message_at_index(store: UpstashChatStore) -> None:
    await store.aset_messages("k", _msgs("a", "c"))
    await store.async_add_message("k", ChatMessage(role="user", content="b"), idx=1)
    assert [m.content for m in await store.aget_messages("k")] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_legacy_async_names_still_work(store: UpstashChatStore) -> None:
    await store.async_set_messages("k", _msgs("a", "b"))
    assert [m.content for m in await store.async_get_messages("k")] == ["a", "b"]
    deleted = await store.async_delete_message("k", 0)
    assert deleted is not None and deleted.content == "a"
    assert await store.async_get_keys() == ["k"]
    last = await store.async_delete_last_message("k")
    assert last is not None and last.content == "b"
    await store.async_delete_messages("k")
    assert await store.async_get_messages("k") == []
