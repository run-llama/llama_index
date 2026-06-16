"""Regression tests for node-id prefix handling in add/async_add (#21483)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis as RedisAsync

from llama_index.core.schema import TextNode
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.base import VECTOR_FIELD_NAME

PREFIX = "semantic_cache_doc"
SEPARATOR = ":"
# The node id begins with 'e', a character that also appears in PREFIX. The old
# ``key.strip(PREFIX + SEPARATOR)`` implementation removed every leading/trailing
# character contained in the prefix, so it would mangle this id into
# "7b95ae7-..."; ``removeprefix`` strips the exact substring instead.
NODE_ID = "e7b95ae7-6369-404d-8287-1f4504121563"


def _make_store() -> RedisVectorStore:
    # Passing only an async client avoids create_index()/any connection in __init__.
    return RedisVectorStore(
        redis_client_async=RedisAsync.from_url("redis://localhost:6379")
    )


def _fake_index(load_return, is_async: bool) -> MagicMock:
    index = MagicMock()
    index.name = "test_index"
    index.prefix = PREFIX
    index.key_separator = SEPARATOR
    index.schema.fields = {
        VECTOR_FIELD_NAME: SimpleNamespace(attrs=SimpleNamespace(dims=4))
    }
    loader = AsyncMock if is_async else MagicMock
    index.load = loader(return_value=load_return)
    return index


def _node() -> TextNode:
    node = TextNode(text="x", id_=NODE_ID)
    node.embedding = [0.1, 0.2, 0.3, 0.4]
    return node


def test_add_returns_exact_node_id():
    """add() must return the original node id, not a strip()-mangled one."""
    store = _make_store()
    store._index = _fake_index([f"{PREFIX}{SEPARATOR}{NODE_ID}"], is_async=False)
    assert store.add([_node()]) == [NODE_ID]


@pytest.mark.asyncio
async def test_async_add_returns_exact_node_id():
    """async_add() must return the original node id."""
    with patch.object(RedisVectorStore, "async_index_exists", new_callable=AsyncMock):
        store = _make_store()
        store._async_index = _fake_index(
            [f"{PREFIX}{SEPARATOR}{NODE_ID}"], is_async=True
        )
        assert await store.async_add([_node()]) == [NODE_ID]
