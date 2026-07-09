import asyncio

from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.redis import RedisKVStore


def test_class():
    names_of_base_classes = [b.__name__ for b in RedisKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


class _SyncHScanIter:
    """Minimal stand-in for redis-py's sync hscan_iter, yielding fixed (key, value) pairs."""

    def __init__(self, items):
        self._items = items

    def __call__(self, name):
        return iter(self._items)


class _AsyncHScanIter:
    """Minimal stand-in for redis-py's async hscan_iter, yielding fixed (key, value) pairs."""

    def __init__(self, items):
        self._items = items

    def __call__(self, name):
        return self._aiter(self._items)

    @staticmethod
    async def _aiter(items):
        for item in items:
            yield item


def _make_kvstore(items):
    """
    Build a RedisKVStore backed by fake sync/async clients that both yield `items`
    from hscan_iter, without touching a real Redis server.
    """
    from unittest.mock import MagicMock

    redis_client = MagicMock()
    redis_client.hscan_iter = _SyncHScanIter(items)
    async_redis_client = MagicMock()
    async_redis_client.hscan_iter = _AsyncHScanIter(items)
    return RedisKVStore(redis_client=redis_client, async_redis_client=async_redis_client)


def test_get_all_with_bytes_keys():
    """
    Regression test for https://github.com/run-llama/llama_index/issues/22115.

    With the redis-py default (decode_responses=False), hscan_iter yields keys as
    bytes, and get_all must decode them.
    """
    items = [(b"doc-1", b'{"a": 1}'), (b"doc-2", b'{"b": 2}')]
    store = _make_kvstore(items)
    assert store.get_all() == {"doc-1": {"a": 1}, "doc-2": {"b": 2}}


def test_get_all_with_str_keys():
    """
    Regression test for https://github.com/run-llama/llama_index/issues/22115.

    With decode_responses=True, hscan_iter yields keys already decoded as str;
    get_all must not call .decode() on them.
    """
    items = [("doc-1", '{"a": 1}'), ("doc-2", '{"b": 2}')]
    store = _make_kvstore(items)
    assert store.get_all() == {"doc-1": {"a": 1}, "doc-2": {"b": 2}}


def test_aget_all_with_bytes_keys():
    """Async counterpart of test_get_all_with_bytes_keys."""
    items = [(b"doc-1", b'{"a": 1}'), (b"doc-2", b'{"b": 2}')]
    store = _make_kvstore(items)
    result = asyncio.run(store.aget_all())
    assert result == {"doc-1": {"a": 1}, "doc-2": {"b": 2}}


def test_aget_all_with_str_keys():
    """Async counterpart of test_get_all_with_str_keys."""
    items = [("doc-1", '{"a": 1}'), ("doc-2", '{"b": 2}')]
    store = _make_kvstore(items)
    result = asyncio.run(store.aget_all())
    assert result == {"doc-1": {"a": 1}, "doc-2": {"b": 2}}
