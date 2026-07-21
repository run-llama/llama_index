from unittest.mock import MagicMock, AsyncMock

import pytest

from llama_index.storage.kvstore.redis import RedisKVStore


def test_get_all_decode_responses_false():
    """Test get_all when Redis returns bytes keys (decode_responses=False)."""
    mock_redis = MagicMock()
    # Simulate hscan_iter returning bytes keys (default behavior)
    mock_redis.hscan_iter.return_value = iter(
        [
            (b"key1", '{"name": "alice"}'),
            (b"key2", '{"name": "bob"}'),
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
            ("key1", '{"name": "alice"}'),
            ("key2", '{"name": "bob"}'),
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
            (b"bytes_key", '{"type": "bytes"}'),
            ("str_key", '{"type": "str"}'),
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
        for item in [(b"akey1", '{"id": 1}'), (b"akey2", '{"id": 2}')]:
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
        for item in [("akey1", '{"id": 1}'), ("akey2", '{"id": 2}')]:
            yield item

    mock_async_redis.hscan_iter.return_value = async_iter()

    store = RedisKVStore(async_redis_client=mock_async_redis)
    result = await store.aget_all()

    assert result == {"akey1": {"id": 1}, "akey2": {"id": 2}}
