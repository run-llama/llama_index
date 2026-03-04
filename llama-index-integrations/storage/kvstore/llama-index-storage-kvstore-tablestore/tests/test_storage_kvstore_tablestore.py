import os

import pytest
from llama_index.core.storage.kvstore.types import BaseKVStore

from llama_index.storage.kvstore.tablestore import TablestoreKVStore


def test_class():
    names_of_base_classes = [b.__name__ for b in TablestoreKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes
    assert TablestoreKVStore.__name__ in names_of_base_classes


def get_tablestore() -> TablestoreKVStore:
    """Test end to end construction and search."""
    # noinspection DuplicatedCode
    end_point = os.getenv("tablestore_end_point")
    instance_name = os.getenv("tablestore_instance_name")
    access_key_id = os.getenv("tablestore_access_key_id")
    access_key_secret = os.getenv("tablestore_access_key_secret")
    if (
        end_point is None
        or instance_name is None
        or access_key_id is None
        or access_key_secret is None
    ):
        pytest.skip(
            "end_point is None or instance_name is None or "
            "access_key_id is None or access_key_secret is None"
        )

    # 1. create tablestore vector store
    store = TablestoreKVStore(
        endpoint=os.getenv("tablestore_end_point"),
        instance_name=os.getenv("tablestore_instance_name"),
        access_key_id=os.getenv("tablestore_access_key_id"),
        access_key_secret=os.getenv("tablestore_access_key_secret"),
    )
    store.delete_all()
    return store


def test_kvstore_basic() -> None:
    test_key = "test_key_basic"
    test_dict = {"a": "1", "b": "2", "c": "3"}
    kvstore = get_tablestore()

    test_collection = "llama_index_kv_store_test_collection"

    kvstore.put(test_key, test_dict, collection=test_collection)

    tables = kvstore._update_collection()
    assert test_collection in tables

    blob = kvstore.get(test_key, collection=test_collection)
    assert blob == test_dict

    blob = kvstore.get(test_key, collection="non_existent")
    assert blob is None

    deleted = kvstore.delete(test_key, collection=test_collection)
    assert deleted


def test_kvstore_delete() -> None:
    test_key = "test_key_delete"
    test_dict = {"a": "b", "c": 1}
    kvstore = get_tablestore()
    kvstore.put(test_key, test_dict)
    d = kvstore.get(test_key)
    assert d == test_dict

    kvstore.delete(test_key)
    blob = kvstore.get(test_key)
    assert blob is None

    kvstore.delete("not_existent")


def test_kvstore_get_all() -> None:
    kvstore = get_tablestore()
    test_key_1 = "ots_test_key_1"
    test_blob_1 = {
        "ots_test_obj_key": "ots_test_obj_val",
        "nested": {
            "n1": 1,
            "n2": "2",
            "n3": True,
        },
        "list": [1, 2, 3],
    }
    kvstore.put(test_key_1, test_blob_1)
    blob = kvstore.get(test_key_1)
    assert blob == test_blob_1
    test_key_2 = "ots_test_key_2"
    test_blob_2 = {"ots_test_obj_key": "test_obj_val"}
    kvstore.put(test_key_2, test_blob_2)
    blob = kvstore.get(test_key_2)
    assert blob == test_blob_2

    blob = kvstore.get_all()
    assert len(blob) == 2
    assert blob[test_key_1] == test_blob_1
    assert blob[test_key_2] == test_blob_2

    kvstore.delete(test_key_1)
    kvstore.delete(test_key_2)


def test_kvstore_put_all() -> None:
    kvstore = get_tablestore()
    test_key = "ots_test_key_put_all_1"
    test_blob = {"ots_test_obj_key": "ots_test_obj_val", "f": 1, "b": True}
    test_key2 = "ots_test_key_put_all_2"
    test_blob2 = {"ots_test_obj_key2": "ots_test_obj_val2", "f": 0.5, "b": False}
    kvstore.put_all([(test_key, test_blob), (test_key2, test_blob2)])
    blob = kvstore.get(test_key)
    assert blob == test_blob
    blob = kvstore.get(test_key2)
    assert blob == test_blob2

    kvstore.delete(test_key)
    kvstore.delete(test_key2)


def test_delete_all() -> None:
    kvstore = get_tablestore()
    kvstore.put("1", {"a": 1})
    kvstore.put("2", {"a": 2})
    kvstore.put("3", {"a": 3})
    kvstore.put("4", {"a": 4})
    kvstore.put("5", {"a": 5})

    kvstore.delete_all()
    assert len(kvstore.get_all()) == 0
