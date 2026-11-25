from pathlib import Path

import pytest
from fsspec.implementations.memory import MemoryFileSystem
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


def test_kvstore_dict(kvstore_with_data: SimpleKVStore) -> None:
    """Test kvstore dict."""
    save_dict = kvstore_with_data.to_dict()
    loaded_kvstore = SimpleKVStore.from_dict(save_dict)
    assert len(loaded_kvstore.get_all()) == 1


@pytest.mark.parametrize("use_memory_fs", [False, True], ids=["local", "memory"])
def test_kvstore_persist_collection_isolation(
    tmp_path: Path, use_memory_fs: bool
) -> None:
    """Persisted collections retain distinct values for the same key."""
    kvstore = SimpleKVStore()
    key = "shared_key"
    tenant_values = {
        "tenant_a": {"value": "a"},
        "tenant_b": {"value": "b"},
    }
    default_value = {"value": "default"}
    kvstore.put(key, default_value)
    for collection, value in tenant_values.items():
        kvstore.put(key, value, collection=collection)

    persist_path = str(tmp_path / "kvstore.json")
    fs = MemoryFileSystem() if use_memory_fs else None
    try:
        if fs is None:
            kvstore.persist(persist_path)
            assert Path(persist_path).is_file()
            loaded = SimpleKVStore.from_persist_path(persist_path)
        else:
            kvstore.persist(persist_path, fs=fs)
            assert fs.exists(persist_path)
            assert not Path(persist_path).exists()
            loaded = SimpleKVStore.from_persist_path(persist_path, fs=fs)

        assert loaded.get(key) == default_value
        assert loaded.get_all() == {key: default_value}
        for collection, value in tenant_values.items():
            assert loaded.get(key, collection=collection) == value
            assert loaded.get_all(collection=collection) == {key: value}

        assert loaded.delete(key, collection="tenant_a")
        assert loaded.get(key, collection="tenant_a") is None
        assert loaded.get_all(collection="tenant_a") == {}
        assert loaded.get(key, collection="tenant_b") == tenant_values["tenant_b"]
        assert loaded.get_all(collection="tenant_b") == {key: tenant_values["tenant_b"]}
        assert loaded.get(key) == default_value
        assert loaded.get_all() == {key: default_value}
    finally:
        if fs is not None and fs.exists(str(tmp_path)):
            fs.rm(str(tmp_path), recursive=True)
