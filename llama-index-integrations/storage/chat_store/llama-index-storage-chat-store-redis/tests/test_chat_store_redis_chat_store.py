from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.redis import RedisChatStore

REDIS_KEY = "redis_chat_store_tests"

# Run Redis Open Source locally via Docker:
# docker run --name redis -p 6379:6379 redis:<version> (omit `:version` or use `:latest` for latest)
#
# https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/docker/
try:
    from redis.client import Redis

    redis_client = Redis.from_url("redis://localhost:6379")
    redis_client.info()
    redis_available = True
except (ImportError, Exception):
    redis_available = False

needs_redis_conn = pytest.mark.skipif(
    not redis_available, reason="Redis Open Source not running locally"
)
"""
Tests marked with this decorator require a running instance of Redis. Otherwise, the
test is skipped.
"""


def test_class():
    names_of_base_classes = [b.__name__ for b in RedisChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


def test_aget_client_cluster_without_running_loop_does_not_raise():
    """
    _aget_client() discards the initial non-cluster async client and
    reconnects with a cluster client when the server turns out to be a
    Redis Cluster. Closing the discarded client used to call
    asyncio.create_task() unconditionally, which raises RuntimeError when
    there's no running event loop - the common case, since RedisChatStore()
    is a plain synchronous constructor. It must close the discarded client
    without needing a running loop.
    """
    # RedisChatStore is a pydantic model; __new__ alone leaves it without
    # the pydantic internals _aget_client's unrelated code paths don't need
    # anyway, so call it unbound against a plain mock standing in for self.
    fake_self = MagicMock()
    fake_self._check_for_cluster.return_value = True

    mock_aredis_client = MagicMock()
    mock_aredis_client.close = AsyncMock()
    fake_self._aredis_cluster_client.return_value = MagicMock()

    with (
        patch(
            "llama_index.storage.chat_store.redis.base.AsyncRedis.from_url",
            return_value=mock_aredis_client,
        ),
        patch(
            "llama_index.storage.chat_store.redis.base.Redis.from_url",
            return_value=MagicMock(),
        ),
    ):
        result = RedisChatStore._aget_client(fake_self, "redis://localhost:6379")

    mock_aredis_client.close.assert_called_once()
    assert result is fake_self._aredis_cluster_client.return_value


@pytest.mark.asyncio
async def test_aget_client_cluster_keeps_close_task_alive_until_it_completes():
    """
    Same discard-and-reconnect path as above, but called with a loop already
    running: closing the discarded client is scheduled via
    asyncio.create_task(), which only keeps a *weak* reference to the
    returned Task. With nothing else referencing it, the task can be
    garbage-collected before it ever runs. _aget_client() must keep a
    strong reference (_background_close_tasks) until the task is done.
    """
    from llama_index.storage.chat_store.redis.base import _background_close_tasks

    fake_self = MagicMock()
    fake_self._check_for_cluster.return_value = True

    mock_aredis_client = MagicMock()
    mock_aredis_client.close = AsyncMock()
    fake_self._aredis_cluster_client.return_value = MagicMock()

    with (
        patch(
            "llama_index.storage.chat_store.redis.base.AsyncRedis.from_url",
            return_value=mock_aredis_client,
        ),
        patch(
            "llama_index.storage.chat_store.redis.base.Redis.from_url",
            return_value=MagicMock(),
        ),
    ):
        result = RedisChatStore._aget_client(fake_self, "redis://localhost:6379")

    assert len(_background_close_tasks) == 1
    (task,) = tuple(_background_close_tasks)
    await task

    mock_aredis_client.close.assert_called_once()
    assert len(_background_close_tasks) == 0
    assert result is fake_self._aredis_cluster_client.return_value


@pytest.fixture()
def redis_chat_store() -> Generator[RedisChatStore, None, None]:
    chat_store = None
    try:
        chat_store = RedisChatStore(
            redis_url="redis://localhost:6379",
            ttl=300,  # 5 minutes
        )
        yield chat_store
    finally:
        if chat_store:
            chat_store.delete_messages(REDIS_KEY)


@needs_redis_conn
def test_add_message(redis_chat_store: RedisChatStore):
    message = ChatMessage(role=MessageRole.USER, content="test_user_message")
    redis_chat_store.add_message(REDIS_KEY, message)

    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 1
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"


@needs_redis_conn
def test_add_message_with_additional_kwargs(redis_chat_store: RedisChatStore):
    message = ChatMessage(
        role=MessageRole.USER,
        content="test_user_message",
        additional_kwargs={"metadata": {"foo": "bar", "count": 5}},
    )
    redis_chat_store.add_message(REDIS_KEY, message)

    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 1
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"

    assert stored_messages[0].additional_kwargs
    assert "metadata" in stored_messages[0].additional_kwargs
    assert stored_messages[0].additional_kwargs["metadata"]["foo"] == "bar"
    assert stored_messages[0].additional_kwargs["metadata"]["count"] == 5


@needs_redis_conn
@pytest.mark.asyncio
async def test_async_add_message(redis_chat_store: RedisChatStore):
    message = ChatMessage(role=MessageRole.USER, content="async_test_user_message")
    await redis_chat_store.async_add_message(REDIS_KEY, message)

    stored_messages = await redis_chat_store.aget_messages(REDIS_KEY)

    assert len(stored_messages) == 1
    assert stored_messages[0].content == "async_test_user_message"
    assert stored_messages[0].role == "user"


@needs_redis_conn
@pytest.mark.asyncio
async def test_async_add_message_with_additional_kwargs(
    redis_chat_store: RedisChatStore,
):
    message = ChatMessage(
        role=MessageRole.USER,
        content="test_user_message",
        additional_kwargs={"metadata": {"foo": "bar", "count": 5}},
    )
    await redis_chat_store.async_add_message(REDIS_KEY, message)

    stored_messages = await redis_chat_store.aget_messages(REDIS_KEY)

    assert len(stored_messages) == 1
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"

    assert stored_messages[0].additional_kwargs
    assert "metadata" in stored_messages[0].additional_kwargs
    assert stored_messages[0].additional_kwargs["metadata"]["foo"] == "bar"
    assert stored_messages[0].additional_kwargs["metadata"]["count"] == 5


@needs_redis_conn
def test_set_messages(redis_chat_store: RedisChatStore):
    messages = [
        ChatMessage(role=MessageRole.USER, content="test_user_message"),
        ChatMessage(role=MessageRole.SYSTEM, content="test_system_message"),
    ]
    redis_chat_store.set_messages(REDIS_KEY, messages)

    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 2
    # Assert first message
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"
    # Assert second message
    assert stored_messages[1].content == "test_system_message"
    assert stored_messages[1].role == "system"


@needs_redis_conn
@pytest.mark.asyncio
async def test_aset_messages(redis_chat_store: RedisChatStore):
    messages = [
        ChatMessage(role=MessageRole.USER, content="test_user_message"),
        ChatMessage(role=MessageRole.SYSTEM, content="test_system_message"),
    ]
    await redis_chat_store.aset_messages(REDIS_KEY, messages)

    stored_messages = await redis_chat_store.aget_messages(REDIS_KEY)

    assert len(stored_messages) == 2
    # Assert first message
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"
    # Assert second message
    assert stored_messages[1].content == "test_system_message"
    assert stored_messages[1].role == "system"


@needs_redis_conn
def test_delete_messages(redis_chat_store: RedisChatStore):
    messages = [
        ChatMessage(role=MessageRole.USER, content="test_user_message"),
        ChatMessage(role=MessageRole.SYSTEM, content="test_system_message"),
    ]
    redis_chat_store.set_messages(REDIS_KEY, messages)

    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 2

    redis_chat_store.delete_messages(REDIS_KEY)
    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 0
    assert stored_messages == []


@needs_redis_conn
@pytest.mark.asyncio
async def test_adelete_messages(redis_chat_store: RedisChatStore):
    messages = [
        ChatMessage(role=MessageRole.USER, content="test_user_message"),
        ChatMessage(role=MessageRole.SYSTEM, content="test_system_message"),
    ]
    await redis_chat_store.aset_messages(REDIS_KEY, messages)

    stored_messages = await redis_chat_store.aget_messages(REDIS_KEY)

    assert len(stored_messages) == 2

    await redis_chat_store.adelete_messages(REDIS_KEY)
    stored_messages = await redis_chat_store.aget_messages(REDIS_KEY)

    assert len(stored_messages) == 0
    assert stored_messages == []


@needs_redis_conn
def test_delete_message(redis_chat_store: RedisChatStore):
    messages = [
        ChatMessage(role=MessageRole.USER, content="test_user_message"),
        ChatMessage(role=MessageRole.SYSTEM, content="test_system_message"),
    ]
    redis_chat_store.set_messages(REDIS_KEY, messages)

    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 2
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"

    redis_chat_store.delete_message(REDIS_KEY, 0)
    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 1
    assert stored_messages[0].content == "test_system_message"
    assert stored_messages[0].role == "system"


@needs_redis_conn
@pytest.mark.asyncio
async def test_adelete_message(redis_chat_store: RedisChatStore):
    messages = [
        ChatMessage(role=MessageRole.USER, content="test_user_message"),
        ChatMessage(role=MessageRole.SYSTEM, content="test_system_message"),
    ]
    await redis_chat_store.aset_messages(REDIS_KEY, messages)

    stored_messages = await redis_chat_store.aget_messages(REDIS_KEY)

    assert len(stored_messages) == 2
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"

    await redis_chat_store.adelete_message(REDIS_KEY, 0)
    stored_messages = await redis_chat_store.aget_messages(REDIS_KEY)

    assert len(stored_messages) == 1
    assert stored_messages[0].content == "test_system_message"
    assert stored_messages[0].role == "system"


@needs_redis_conn
def test_delete_last_message(redis_chat_store: RedisChatStore):
    messages = [
        ChatMessage(role=MessageRole.USER, content="test_user_message"),
        ChatMessage(role=MessageRole.SYSTEM, content="test_system_message"),
    ]
    redis_chat_store.set_messages(REDIS_KEY, messages)

    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 2
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"

    redis_chat_store.delete_last_message(REDIS_KEY)
    stored_messages = redis_chat_store.get_messages(REDIS_KEY)

    assert len(stored_messages) == 1
    assert stored_messages[0].content == "test_user_message"
    assert stored_messages[0].role == "user"


@needs_redis_conn
def test_get_keys(redis_chat_store: RedisChatStore):
    message = ChatMessage(role=MessageRole.USER, content="test_user_message")
    redis_chat_store.add_message(REDIS_KEY, message)

    keys = redis_chat_store.get_keys()

    assert len(keys) == 1
    assert keys[0] == REDIS_KEY
