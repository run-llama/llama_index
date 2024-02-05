from typing import List

import pytest
from llama_index.storage.kvstore.indexed_mongodb_kvstore import IndexedMongoDBKVStore
from tests.storage.kvstore.mock_mongodb import MockMongoClient

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None  # type: ignore


@pytest.fixture()
def indexed_mongo_collection_name() -> str:
    return "test_collection"


@pytest.fixture()
def indexed_columns() -> List[str]:
    return ["a", "b", "c"]


@pytest.fixture()
def indexed_mongo_kvstore(
    mongo_client: MockMongoClient,
    indexed_mongo_collection_name: str,
    indexed_columns: List[str],
) -> IndexedMongoDBKVStore:
    return IndexedMongoDBKVStore(
        mongo_client=mongo_client,
        collection_name=indexed_mongo_collection_name,
        indexed_attributes=indexed_columns,
    )


@pytest.fixture()
def kvstore_with_data(
    indexed_mongo_kvstore: IndexedMongoDBKVStore,
) -> IndexedMongoDBKVStore:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    indexed_mongo_kvstore.put(test_key, test_blob)
    return indexed_mongo_kvstore


@pytest.mark.skipif(MongoClient is None, reason="pymongo not installed")
def test_kvstore_basic(
    indexed_mongo_kvstore: IndexedMongoDBKVStore, indexed_mongo_collection_name: str
) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    indexed_mongo_kvstore.put(test_key, test_blob)
    blob = indexed_mongo_kvstore.get(test_key)
    assert blob == test_blob


@pytest.mark.skipif(MongoClient is None, reason="pymongo not installed")
def test_kvstore_get_with_filters(
    indexed_mongo_kvstore: IndexedMongoDBKVStore,
    indexed_mongo_collection_name: str,
    indexed_columns: List[str],
) -> None:
    equal_blob_test_keys = ["test_key1", "test_key2", "test_key3"]
    for test_key in equal_blob_test_keys:
        test_blob = {col_name: col_name for col_name in indexed_columns}
        test_blob["value_key"] = test_key
        indexed_mongo_kvstore.put(test_key, test_blob)
    unequal_blob_test_keys = ["test_key4", "test_key5", "test_key6"]
    for test_key in unequal_blob_test_keys:
        test_blob = {col_name: test_key for col_name in indexed_columns}
        test_blob["value_key"] = test_key
        indexed_mongo_kvstore.put(test_key, test_blob)

    # Test filter over equal blobs
    equal_filter = {col_name: col_name for col_name in indexed_columns}
    equal_blobs = indexed_mongo_kvstore.get_all_with_filters(equal_filter)
    assert len(equal_blobs) == len(equal_blob_test_keys)

    # Test filter over unequal blobs
    for unequal_blob_test_key in unequal_blob_test_keys:
        unequal_filter = {
            col_name: unequal_blob_test_key for col_name in indexed_columns
        }
        unequal_blobs = indexed_mongo_kvstore.get_all_with_filters(unequal_filter)
        assert len(unequal_blobs) == 1
        assert unequal_blobs[unequal_blob_test_key] == indexed_mongo_kvstore.get(
            unequal_blob_test_key
        )

    # Test filter over non-indexed column "value_key"
    for test_key in equal_blob_test_keys + unequal_blob_test_keys:
        value_filter = {"value_key": test_key}
        value_blobs = indexed_mongo_kvstore.get_all_with_filters(value_filter)
        assert len(value_blobs) == 1
        assert value_blobs[test_key] == indexed_mongo_kvstore.get(test_key)
