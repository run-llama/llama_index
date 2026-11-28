from pathlib import Path

import fsspec
import pytest
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore


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


@pytest.mark.parametrize("protocol", ["file", "memory"])
def test_persist_serialization_failure_preserves_previous_snapshot(
    tmp_path: Path, protocol: str
) -> None:
    path = str(tmp_path / "nested" / "kvstore.json")
    fs = fsspec.filesystem(protocol)
    store = SimpleKVStore()
    store.put("valid", {"value": "café"})
    store.persist(path, fs=fs)

    store.put("invalid", {"value": object()})
    with pytest.raises(TypeError):
        store.persist(path, fs=fs)

    restored = SimpleKVStore.from_persist_path(path, fs=fs)
    assert restored.get("valid") == {"value": "café"}
    assert restored.get("invalid") is None


def test_kvstore_dict(kvstore_with_data: SimpleKVStore) -> None:
    """Test kvstore dict."""
    save_dict = kvstore_with_data.to_dict()
    loaded_kvstore = SimpleKVStore.from_dict(save_dict)
    assert len(loaded_kvstore.get_all()) == 1
