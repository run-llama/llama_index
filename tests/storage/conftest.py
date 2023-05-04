import pytest
from llama_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore
from llama_index.storage.kvstore.simple_kvstore import SimpleKVStore
from tests.storage.kvstore.mock_mongodb import MockMongoClient


@pytest.fixture()
def mongo_client() -> MockMongoClient:
    return MockMongoClient()


@pytest.fixture()
def mongo_kvstore(mongo_client: MockMongoClient) -> MongoDBKVStore:
    return MongoDBKVStore(mongo_client=mongo_client)  # type: ignore


@pytest.fixture()
def simple_kvstore() -> SimpleKVStore:
    return SimpleKVStore()
