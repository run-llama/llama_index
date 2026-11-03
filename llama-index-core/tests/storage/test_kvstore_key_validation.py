import pytest

from llama_index.core.storage.kvstore.types import MutableMappingKVStore


def _store():
    return MutableMappingKVStore(dict)


@pytest.mark.parametrize("key", [123, 1.5, (1, 2), None])
def test_a_non_string_key_is_named(key):
    store = _store()

    with pytest.raises(ValueError, match="KVStore key must be a string"):
        store.put(key, {"a": 1})


def test_an_unhashable_key_is_named():
    store = _store()

    with pytest.raises(ValueError, match="KVStore key must be a string"):
        store.put(["a"], {"a": 1})


def test_a_string_key_still_roundtrips():
    store = _store()
    store.put("doc-1", {"a": 1})

    assert store.get("doc-1") == {"a": 1}
    assert store.get_all() == {"doc-1": {"a": 1}}
