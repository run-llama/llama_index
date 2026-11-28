from pathlib import Path
from typing import Callable

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


def test_kvstore_dict(kvstore_with_data: SimpleKVStore) -> None:
    """Test kvstore dict."""
    save_dict = kvstore_with_data.to_dict()
    loaded_kvstore = SimpleKVStore.from_dict(save_dict)
    assert len(loaded_kvstore.get_all()) == 1


@pytest.mark.parametrize("restore", [SimpleKVStore, SimpleKVStore.from_dict])
def test_kvstore_import_isolates_input(
    restore: Callable[[dict], SimpleKVStore],
) -> None:
    data = {"data": {"node": {"labels": ["original"]}}}
    store = restore(data)

    data["data"]["node"]["labels"].append("changed")
    data["data"]["extra"] = {"labels": []}

    assert store.get_all() == {"node": {"labels": ["original"]}}


@pytest.mark.parametrize("restore", [SimpleKVStore, SimpleKVStore.from_dict])
def test_kvstore_import_isolates_store(
    restore: Callable[[dict], SimpleKVStore],
) -> None:
    data = {"data": {"node": {"labels": ["original"]}}}
    store = restore(data)

    store.delete("node")
    store.put("extra", {"labels": []})

    assert data == {"data": {"node": {"labels": ["original"]}}}


def test_kvstore_export_isolates_store(simple_kvstore: SimpleKVStore) -> None:
    simple_kvstore.put("node", {"labels": ["original"]})
    snapshot = simple_kvstore.to_dict()

    snapshot["data"]["node"]["labels"].append("changed")
    snapshot["data"]["extra"] = {"labels": []}

    assert simple_kvstore.get_all() == {"node": {"labels": ["original"]}}


def test_kvstore_export_keeps_snapshot(simple_kvstore: SimpleKVStore) -> None:
    simple_kvstore.put("node", {"labels": ["original"]})
    snapshot = simple_kvstore.to_dict()

    simple_kvstore.delete("node")
    simple_kvstore.put("extra", {"labels": []})

    assert snapshot == {"data": {"node": {"labels": ["original"]}}}
