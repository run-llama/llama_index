import pytest
from gpt_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore
from tests.storage.kvstore.mock_mongodb import MockMongoClient


@pytest.fixture()
def mongo_client() -> MockMongoClient:
    return MockMongoClient()


@pytest.fixture()
def kvstore(mongo_client: MockMongoClient) -> MongoDBKVStore:
    return MongoDBKVStore(mongo_client=mongo_client)  # type: ignore


@pytest.fixture()
def kvstore_with_data(kvstore: MongoDBKVStore) -> MongoDBKVStore:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    kvstore.put(test_key, test_blob)
    return kvstore


def test_kvstore_basic(kvstore: MongoDBKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    kvstore.put(test_key, test_blob)
    blob = kvstore.get(test_key)
    assert blob == test_blob

    blob = kvstore.get(test_key, collection="non_existent")
    assert blob == None
