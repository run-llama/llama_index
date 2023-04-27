import pytest
from gpt_index.storage.kvstore.simple_kvstore import SimpleKVStore


@pytest.fixture()
def kvstore_with_data(simple_kvstore: SimpleKVStore) -> SimpleKVStore:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    simple_kvstore.put(test_key, test_blob)
    return simple_kvstore


def test_kvstore_basic(simple_kvstore: SimpleKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    simple_kvstore.put(test_key, test_blob)
    blob = simple_kvstore.get(test_key)
    assert blob == test_blob

    blob = simple_kvstore.get(test_key, collection="non_existent")
    assert blob is None


def test_kvstore_persist(kvstore_with_data: SimpleKVStore) -> None:
    persist_path = kvstore_with_data.persist_path
    kvstore_with_data.persist()

    loaded_kvstore = SimpleKVStore(persist_path)
    assert len(loaded_kvstore.get_all()) == 1
