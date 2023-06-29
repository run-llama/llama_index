import pytest

from llama_index.storage.kvstore.redis_kvstore import RedisKVStore


try:
    from redis import Redis
except ImportError:
    Redis = None  # type: ignore


@pytest.fixture()
def kvstore_with_data(redis_kvstore: RedisKVStore) -> RedisKVStore:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    redis_kvstore.put(test_key, test_blob)
    return redis_kvstore


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_kvstore_basic(redis_kvstore: RedisKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    redis_kvstore.put(test_key, test_blob)
    blob = redis_kvstore.get(test_key)
    assert blob == test_blob

    blob = redis_kvstore.get(test_key, collection="non_existent")
    assert blob is None


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_kvstore_delete(redis_kvstore: RedisKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    redis_kvstore.put(test_key, test_blob)
    blob = redis_kvstore.get(test_key)
    assert blob == test_blob

    redis_kvstore.delete(test_key)
    blob = redis_kvstore.get(test_key)
    assert blob is None


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_kvstore_getall(redis_kvstore: RedisKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    redis_kvstore.put(test_key, test_blob)
    blob = redis_kvstore.get(test_key)
    assert blob == test_blob
    test_key = "test_key_2"
    test_blob = {"test_obj_key": "test_obj_val"}
    redis_kvstore.put(test_key, test_blob)
    blob = redis_kvstore.get(test_key)
    assert blob == test_blob

    blob = redis_kvstore.get_all()
    assert len(blob) == 2
