import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.kvstore.redis.base import _normalize_redis_key


def test_class():
    names_of_base_classes = [b.__name__ for b in RedisKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


def test_normalize_redis_key_bytes():
    assert _normalize_redis_key(b"doc-1") == "doc-1"


def test_normalize_redis_key_str():
    # decode_responses=True returns str keys — must not call .decode()
    assert _normalize_redis_key("doc-1") == "doc-1"


def test_get_all_decode_responses_false():
    """Test get_all when Redis returns bytes keys (decode_responses=False)."""
    mock_redis = MagicMock()
    # Simulate hscan_iter returning bytes keys (default behavior)
    mock_redis.hscan_iter.return_value = iter(
        [
            (b"key1", json.dumps({"name": "alice"}).encode("utf-8")),
            (b"key2", json.dumps({"name": "bob"}).encode("utf-8")),
        ]
    )

    store = RedisKVStore(redis_client=mock_redis)
    result = store.get_all()

    assert result == {"key1": {"name": "alice"}, "key2": {"name": "bob"}}
    mock_redis.hscan_iter.assert_called_once()


def test_get_all_decode_responses_true():
    """Test get_all when Redis returns string keys (decode_responses=True)."""
    mock_redis = MagicMock()
    # Simulate hscan_iter returning string keys (decode_responses=True)
    mock_redis.hscan_iter.return_value = iter(
        [
            ("key1", json.dumps({"name": "alice"})),
            ("key2", json.dumps({"name": "bob"})),
        ]
    )

    store = RedisKVStore(redis_client=mock_redis)
    result = store.get_all()

    assert result == {"key1": {"name": "alice"}, "key2": {"name": "bob"}}
    mock_redis.hscan_iter.assert_called_once()


def test_get_all_mixed_keys():
    """Test get_all with mixed bytes/string keys (edge case sanity check)."""
    mock_redis = MagicMock()
    mock_redis.hscan_iter.return_value = iter(
        [
            (b"bytes_key", json.dumps({"type": "bytes"}).encode("utf-8")),
            ("str_key", json.dumps({"type": "str"})),
        ]
    )

    store = RedisKVStore(redis_client=mock_redis)
    result = store.get_all()

    assert result == {
        "bytes_key": {"type": "bytes"},
        "str_key": {"type": "str"},
    }


@pytest.mark.asyncio
async def test_aget_all_decode_responses_false():
    """Test aget_all when async Redis returns bytes keys (decode_responses=False)."""
    mock_async_redis = AsyncMock()

    # Simulate async hscan_iter returning bytes keys
    async def async_iter():
        for item in [
            (b"akey1", json.dumps({"id": 1}).encode("utf-8")),
            (b"akey2", json.dumps({"id": 2}).encode("utf-8")),
        ]:
            yield item

    mock_async_redis.hscan_iter.return_value = async_iter()

    store = RedisKVStore(async_redis_client=mock_async_redis)
    result = await store.aget_all()

    assert result == {"akey1": {"id": 1}, "akey2": {"id": 2}}


@pytest.mark.asyncio
async def test_aget_all_decode_responses_true():
    """Test aget_all when async Redis returns string keys (decode_responses=True)."""
    mock_async_redis = AsyncMock()

    # Simulate async hscan_iter returning string keys
    async def async_iter():
        for item in [
            ("akey1", json.dumps({"id": 1})),
            ("akey2", json.dumps({"id": 2})),
        ]:
            yield item

    mock_async_redis.hscan_iter.return_value = async_iter()

    store = RedisKVStore(async_redis_client=mock_async_redis)
    result = await store.aget_all()

    assert result == {"akey1": {"id": 1}, "akey2": {"id": 2}}
