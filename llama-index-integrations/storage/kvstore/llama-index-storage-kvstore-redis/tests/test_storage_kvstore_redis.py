import asyncio
import json

from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.redis import RedisKVStore


def test_class():
    names_of_base_classes = [b.__name__ for b in RedisKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


class MockRedisClient:
    def __init__(self, entries):
        self.entries = entries

    def hscan_iter(self, name):
        return iter(self.entries)


class MockAsyncRedisClient:
    def __init__(self, entries):
        self.entries = entries

    async def hscan_iter(self, name):
        for entry in self.entries:
            yield entry


def test_get_all_decodes_byte_keys():
    redis_client = MockRedisClient([(b"doc1", json.dumps({"hash": "abc"}).encode())])
    kvstore = RedisKVStore(
        redis_client=redis_client,
        async_redis_client=MockAsyncRedisClient([]),
    )

    assert kvstore.get_all() == {"doc1": {"hash": "abc"}}


def test_get_all_accepts_decoded_string_keys():
    redis_client = MockRedisClient([("doc1", json.dumps({"hash": "abc"}))])
    kvstore = RedisKVStore(
        redis_client=redis_client,
        async_redis_client=MockAsyncRedisClient([]),
    )

    assert kvstore.get_all() == {"doc1": {"hash": "abc"}}


def test_aget_all_accepts_decoded_string_keys():
    async def _run_test():
        kvstore = RedisKVStore(
            redis_client=MockRedisClient([]),
            async_redis_client=MockAsyncRedisClient(
                [("doc1", json.dumps({"hash": "abc"}))]
            ),
        )

        assert await kvstore.aget_all() == {"doc1": {"hash": "abc"}}

    asyncio.run(_run_test())
