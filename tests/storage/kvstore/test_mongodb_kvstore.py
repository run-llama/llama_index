import pytest
from gpt_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore


@pytest.fixture()
def kvstore_with_data(mongo_kvstore: MongoDBKVStore) -> MongoDBKVStore:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    mongo_kvstore.put(test_key, test_blob)
    return mongo_kvstore


def test_kvstore_basic(mongo_kvstore: MongoDBKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    mongo_kvstore.put(test_key, test_blob)
    blob = mongo_kvstore.get(test_key)
    assert blob == test_blob

    blob = mongo_kvstore.get(test_key, collection="non_existent")
    assert blob is None
