from pathlib import Path
import pytest
from gpt_index.storage.kvstore.simple_kvstore import SimpleKVStore


@pytest.fixture()
def kvstore(tmp_path: Path) -> SimpleKVStore:
    file_path = str(tmp_path / "test_file.txt")
    return SimpleKVStore(file_path)


@pytest.fixture()
def kvstore_with_data(kvstore: SimpleKVStore) -> SimpleKVStore:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    kvstore.put(test_key, test_blob)
    return kvstore


def test_kvstore_basic(kvstore: SimpleKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    kvstore.put(test_key, test_blob)
    blob = kvstore.get(test_key)
    assert blob == test_blob

    blob = kvstore.get(test_key, collection="non_existent")
    assert blob == None


def test_kvstore_persist(kvstore_with_data: SimpleKVStore) -> None:
    persist_path = kvstore_with_data.persist_path
    kvstore_with_data.persist()

    loaded_kvstore = SimpleKVStore(persist_path)
    assert len(loaded_kvstore.get_all()) == 1
