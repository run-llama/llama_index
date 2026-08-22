"""
Regression tests for node ID round-tripping in add/async_add.

Node IDs whose characters intersect the index prefix alphabet were
corrupted when extracted from Redis keys with str.strip(); the exact
prefix must be removed with str.removeprefix() instead.

These tests use fakes, so no live Redis instance is required.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.schema import TextNode
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import RedisVectorStoreSchema

# Node IDs chosen so their leading/trailing characters intersect the
# default prefix alphabet ("llama_index/vector"): e.g. "doc..." would be
# eaten from the front by strip(), and a trailing "...vector" from the back.
NODE_IDS = [
    "doc1234567890",
    "e7b95ae7-6369-404d-8287-1f4504121563",
    "node-id-vector",
]


def _build_nodes() -> list:
    embedding = [0.0] * 1536
    return [
        TextNode(text=f"test {i}", id_=node_id, embedding=embedding)
        for i, node_id in enumerate(NODE_IDS)
    ]


@pytest.fixture()
def vector_store():
    store = RedisVectorStore(
        schema=RedisVectorStoreSchema(),
        redis_client_async=MagicMock(),
    )
    # Skip the async index existence check, which requires a live Redis.
    store.created_async_index = True
    return store


def _fake_keys(vector_store, nodes):
    prefix = vector_store._index.prefix
    separator = vector_store._index.key_separator
    return [f"{prefix}{separator}{node.node_id}" for node in nodes]


def _mock_sync_load(vector_store, nodes):
    keys = _fake_keys(vector_store, nodes)
    vector_store._index.load = MagicMock(return_value=keys)


def _mock_async_load(vector_store, nodes):
    keys = _fake_keys(vector_store, nodes)
    vector_store._async_index.load = AsyncMock(return_value=keys)


def test_add_returns_uncorrupted_node_ids(vector_store):
    nodes = _build_nodes()
    _mock_sync_load(vector_store, nodes)

    returned_ids = vector_store.add(nodes)

    assert returned_ids == NODE_IDS


@pytest.mark.asyncio
async def test_async_add_returns_uncorrupted_node_ids(vector_store):
    nodes = _build_nodes()
    _mock_async_load(vector_store, nodes)

    returned_ids = await vector_store.async_add(nodes)

    assert returned_ids == NODE_IDS
