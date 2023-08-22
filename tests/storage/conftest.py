import pytest
from llama_index.storage.kvstore.firestore_kvstore import FirestoreKVStore
from llama_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore
from llama_index.storage.kvstore.redis_kvstore import RedisKVStore
from llama_index.storage.kvstore.simple_kvstore import SimpleKVStore
from tests.storage.kvstore.mock_mongodb import MockMongoClient


@pytest.fixture()
def mongo_client() -> MockMongoClient:
    return MockMongoClient()


@pytest.fixture()
def mongo_kvstore(mongo_client: MockMongoClient) -> MongoDBKVStore:
    return MongoDBKVStore(mongo_client=mongo_client)  # type: ignore


@pytest.fixture()
def firestore_kvstore() -> FirestoreKVStore:
    return FirestoreKVStore()


@pytest.fixture()
def simple_kvstore() -> SimpleKVStore:
    return SimpleKVStore()


@pytest.fixture()
def redis_kvstore() -> "RedisKVStore":
    try:
        from redis import Redis

        client = Redis.from_url(url="redis://127.0.0.1:6379")
    except ImportError:
        return RedisKVStore(redis_client=None, redis_url="redis://127.0.0.1:6379")
    return RedisKVStore(redis_client=client)
