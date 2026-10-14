from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from redis import Redis
from redis.asyncio import Redis as RedisAsync
from redisvl.schema import IndexSchema

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import (
    RedisVectorStoreSchema,
    VECTOR_FIELD_NAME,
)


@pytest.fixture(params=["FLOAT32", "FLOAT64"])
def store_and_indexes(request):
    schema_dict = RedisVectorStoreSchema().to_dict()
    vector_field = next(
        field for field in schema_dict["fields"] if field["name"] == VECTOR_FIELD_NAME
    )
    vector_field["attrs"].update(dims=3, datatype=request.param)
    schema = IndexSchema.from_dict(schema_dict)
    sync_index = MagicMock(schema=schema)
    sync_index.load.return_value = []
    async_index = MagicMock(schema=schema)
    async_index.create = AsyncMock()
    async_index.load = AsyncMock(return_value=[])
    async_index.query = AsyncMock(return_value=[])
    sync_index.query.return_value = []

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
            redis_client=Redis.from_url("redis://localhost:6379"),
            redis_client_async=RedisAsync.from_url("redis://localhost:6379"),
        )
    return store, sync_index, async_index, request.param.lower()


def _assert_embedding_buffer(buffer, dtype):
    assert len(buffer) == 3 * np.dtype(dtype).itemsize
    np.testing.assert_array_equal(np.frombuffer(buffer, dtype=dtype), [0.25, -0.5, 1.0])


def test_add_serializes_embedding_with_schema_datatype(store_and_indexes):
    store, sync_index, _, dtype = store_and_indexes
    store.add([TextNode(text="test", embedding=[0.25, -0.5, 1.0])])
    record = sync_index.load.call_args.args[0][0]
    _assert_embedding_buffer(record[VECTOR_FIELD_NAME], dtype)


@pytest.mark.asyncio
async def test_async_add_serializes_embedding_with_schema_datatype(store_and_indexes):
    store, _, async_index, dtype = store_and_indexes
    await store.async_add([TextNode(text="test", embedding=[0.25, -0.5, 1.0])])
    record = async_index.load.call_args.args[0][0]
    _assert_embedding_buffer(record[VECTOR_FIELD_NAME], dtype)


def test_query_serializes_embedding_with_schema_datatype(store_and_indexes):
    store, sync_index, _, dtype = store_and_indexes
    store.query(VectorStoreQuery(query_embedding=[0.25, -0.5, 1.0]))
    query = sync_index.query.call_args.args[0]
    _assert_embedding_buffer(query.params[query.VECTOR_PARAM], dtype)


@pytest.mark.asyncio
async def test_async_query_serializes_embedding_with_schema_datatype(store_and_indexes):
    store, _, async_index, dtype = store_and_indexes
    await store.aquery(VectorStoreQuery(query_embedding=[0.25, -0.5, 1.0]))
    query = async_index.query.call_args.args[0]
    _assert_embedding_buffer(query.params[query.VECTOR_PARAM], dtype)
