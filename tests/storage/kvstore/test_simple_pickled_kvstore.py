import pytest
from llama_index.storage.kvstore.simple_pickled_kvstore import SimplePickledKVStore
from pathlib import Path


@pytest.fixture()
def pickled_kvstore_with_data(simple_pickled_kvstore: SimplePickledKVStore) -> SimplePickledKVStore:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    simple_pickled_kvstore.put(test_key, test_blob)
    return simple_pickled_kvstore


def test_kvstore_basic(pickled_kvstore_with_data: SimplePickledKVStore) -> None:
    test_key = "test_key"
    test_blob = {"test_obj_key": "test_obj_val"}
    pickled_kvstore_with_data.put(test_key, test_blob)
    blob = pickled_kvstore_with_data.get(test_key)
    assert blob == test_blob

    blob = pickled_kvstore_with_data.get(test_key, collection="non_existent")
    assert blob is None


def test_kvstore_persist(tmp_path: Path, pickled_kvstore_with_data: SimplePickledKVStore) -> None:
    """Test kvstore persist."""
    testpath = str(Path(tmp_path) / "pickled_kvstore.pkl")
    pickled_kvstore_with_data.persist(testpath)
    loaded_kvstore = SimplePickledKVStore.from_persist_path(testpath)
    assert len(loaded_kvstore.get_all()) == 1


def test_kvstore_dict(pickled_kvstore_with_data: SimplePickledKVStore) -> None:
    """Test kvstore dict."""
    save_dict = pickled_kvstore_with_data.to_dict()
    loaded_kvstore = SimplePickledKVStore.from_dict(save_dict)
    assert len(loaded_kvstore.get_all()) == 1
