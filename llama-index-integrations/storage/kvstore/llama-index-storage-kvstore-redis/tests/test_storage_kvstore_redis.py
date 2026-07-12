import asyncio
import json
from typing import AsyncIterator, Iterator, Tuple
from unittest.mock import MagicMock

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


def test_get_all_with_bytes_keys():
    """Default Redis client (decode_responses=False) yields bytes keys."""
    mock_client = MagicMock()

    def hscan_iter(name: str) -> Iterator[Tuple[bytes, bytes]]:
        yield b"key1", json.dumps({"a": 1}).encode("utf-8")
        yield b"key2", json.dumps({"b": 2}).encode("utf-8")

    mock_client.hscan_iter.side_effect = hscan_iter

    store = RedisKVStore(
        redis_client=mock_client,
        async_redis_client=MagicMock(),
    )
    result = store.get_all(collection="test")

    assert result == {"key1": {"a": 1}, "key2": {"b": 2}}
    mock_client.hscan_iter.assert_called_once_with(name="test")


def test_get_all_with_str_keys_decode_responses_true():
    """
    Redis client with decode_responses=True yields str keys.

    Unconditionally calling key.decode() previously raised AttributeError
    (GitHub issue #22115).
    """
    mock_client = MagicMock()

    def hscan_iter(name: str) -> Iterator[Tuple[str, str]]:
        yield "key1", json.dumps({"a": 1})
        yield "key2", json.dumps({"b": 2})

    mock_client.hscan_iter.side_effect = hscan_iter

    store = RedisKVStore(
        redis_client=mock_client,
        async_redis_client=MagicMock(),
    )
    result = store.get_all(collection="test")

    assert result == {"key1": {"a": 1}, "key2": {"b": 2}}


def test_aget_all_with_str_keys_decode_responses_true():
    mock_async_client = MagicMock()

    async def hscan_iter(name: str) -> AsyncIterator[Tuple[str, str]]:
        yield "async-key", json.dumps({"x": 9})

    mock_async_client.hscan_iter.side_effect = hscan_iter

    store = RedisKVStore(
        redis_client=MagicMock(),
        async_redis_client=mock_async_client,
    )
    result = asyncio.run(store.aget_all(collection="async-test"))

    assert result == {"async-key": {"x": 9}}


def test_aget_all_with_bytes_keys():
    mock_async_client = MagicMock()

    async def hscan_iter(name: str) -> AsyncIterator[Tuple[bytes, bytes]]:
        yield b"async-key", json.dumps({"x": 9}).encode("utf-8")

    mock_async_client.hscan_iter.side_effect = hscan_iter

    store = RedisKVStore(
        redis_client=MagicMock(),
        async_redis_client=mock_async_client,
    )
    result = asyncio.run(store.aget_all(collection="async-test"))

    assert result == {"async-key": {"x": 9}}
