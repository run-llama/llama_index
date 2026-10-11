from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from llama_index.core.schema import TextNode
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import RedisVectorStoreSchema


@pytest.fixture(params=[False, True], ids=["keep-index", "overwrite-index"])
def store_and_index(request):
    schema = RedisVectorStoreSchema()
    schema.fields["vector"].attrs.dims = 3
    sync_index = MagicMock(schema=schema)
    async_index = MagicMock(schema=schema)
    async_index.create = AsyncMock()
    async_index.load = AsyncMock(return_value=[])
    with (
        patch(
            "llama_index.vector_stores.redis.base.SearchIndex", return_value=sync_index
        ),
        patch(
            "llama_index.vector_stores.redis.base.AsyncSearchIndex",
            return_value=async_index,
        ),
    ):
        store = RedisVectorStore(
            schema=schema,
            redis_client_async=Redis.from_url("redis://localhost:6379"),
            overwrite=request.param,
        )
    return store, sync_index, async_index


@pytest.mark.asyncio
@pytest.mark.parametrize("already_created", [False, True])
async def test_empty_async_add_does_not_initialize_or_overwrite_index(
    store_and_index, already_created
):
    store, _, index = store_and_index
    store.created_async_index = already_created

    assert await store.async_add([]) == []

    index.create.assert_not_awaited()
    index.load.assert_not_awaited()
    assert store.created_async_index is already_created


def test_empty_sync_add_remains_a_noop(store_and_index):
    store, sync_index, async_index = store_and_index

    assert store.add([]) == []

    sync_index.create.assert_not_called()
    sync_index.load.assert_not_called()
    async_index.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_created", [False, True])
async def test_nonempty_async_add_still_initializes_and_loads(
    store_and_index, already_created
):
    store, _, index = store_and_index
    store.created_async_index = already_created

    await store.async_add([TextNode(text="test", embedding=[0.25, -0.5, 1.0])])

    if already_created:
        index.create.assert_not_awaited()
    elif store._overwrite:
        index.create.assert_awaited_once_with(overwrite=True, drop=True)
    else:
        index.create.assert_awaited_once_with()
    index.load.assert_awaited_once()
    assert store.created_async_index
