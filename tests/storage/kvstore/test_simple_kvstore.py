import pytest
from llama_index.storage.kvstore.simple_kvstore import SimpleKVStore
from pathlib import Path


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


def test_kvstore_persist(tmp_path: Path, kvstore_with_data: SimpleKVStore) -> None:
    """Test kvstore persist."""
    testpath = str(Path(tmp_path) / "kvstore.json")
    kvstore_with_data.persist(testpath)
    loaded_kvstore = SimpleKVStore.from_persist_path(testpath)
    assert len(loaded_kvstore.get_all()) == 1


def test_kvstore_dict(kvstore_with_data: SimpleKVStore) -> None:
    """Test kvstore dict."""
    save_dict = kvstore_with_data.to_dict()
    loaded_kvstore = SimpleKVStore.from_dict(save_dict)
    assert len(loaded_kvstore.get_all()) == 1
