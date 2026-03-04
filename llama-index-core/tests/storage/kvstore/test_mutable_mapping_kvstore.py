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
