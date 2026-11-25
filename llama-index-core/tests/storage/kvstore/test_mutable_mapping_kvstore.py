import pytest

from llama_index.core.storage.kvstore.types import MutableMappingKVStore
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore


def test_simple_kvstore():
    kv_store = SimpleKVStore()
    assert isinstance(kv_store, MutableMappingKVStore)
    kv_store.put(key="foo", val={"foo": "bar"})
    assert kv_store.get_all() == {"foo": {"foo": "bar"}}


def test_sync_methods():
    mut_mapping = MutableMappingKVStore(dict)
    mut_mapping.put(key="foo", val={"foo": "bar"})
    assert mut_mapping.get("foo") == {"foo": "bar"}
    assert mut_mapping.get("bar") is None
    mut_mapping.put(key="bar", val={"bar": "foo"})
    assert mut_mapping.get_all() == {"foo": {"foo": "bar"}, "bar": {"bar": "foo"}}
    mut_mapping.delete(key="bar")
    assert mut_mapping.get_all() == {"foo": {"foo": "bar"}}


@pytest.mark.asyncio
async def test_async_methods():
    mut_mapping = MutableMappingKVStore(dict)
    await mut_mapping.aput(key="foo", val={"foo": "bar"})
    assert await mut_mapping.aget("foo") == {"foo": "bar"}
    assert await mut_mapping.aget("bar") is None
    await mut_mapping.aput(key="bar", val={"bar": "foo"})
    assert await mut_mapping.aget_all() == {
        "foo": {"foo": "bar"},
        "bar": {"bar": "foo"},
    }
    await mut_mapping.adelete(key="bar")
    assert await mut_mapping.aget_all() == {"foo": {"foo": "bar"}}


def test_put_all_batch_and_collection_isolation():
    kv_store = SimpleKVStore()
    assert kv_store.get("missing", collection="nonexistent") is None
    assert kv_store.get_all(collection="nonexistent") == {}
    assert kv_store.delete("missing", collection="nonexistent") is False
    assert kv_store.to_dict() == {}

    val_a = {"a": 1}
    val_b = {"b": 2}
    kv_store.put_all([("k1", val_a), ("k2", val_b)], collection="col", batch_size=10)
    val_a["a"] = 99

    all_vals = kv_store.get_all(collection="col")
    assert all_vals == {"k1": {"a": 1}, "k2": {"b": 2}}
    all_vals["k1"]["a"] = 42
    assert kv_store.get("k1", collection="col") == {"a": 1}


@pytest.mark.asyncio
async def test_aput_all_batch():
    mut_mapping = MutableMappingKVStore(dict)
    await mut_mapping.aput_all(
        [("k1", {"v": 1}), ("k2", {"v": 2})], collection="col", batch_size=8
    )
    assert await mut_mapping.aget_all(collection="col") == {
        "k1": {"v": 1},
        "k2": {"v": 2},
    }
