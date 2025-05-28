import cachetools
import pytest

from llama_index.storage.kvstore.cachetools import CachetoolsKVStore


def test_class():
    names_of_base_classes = [b.__name__ for b in CachetoolsKVStore.__mro__]
    assert CachetoolsKVStore.__name__ in names_of_base_classes


def test_cache():
    cache = CachetoolsKVStore(cachetools.LRUCache, maxsize=2)

    assert len(cache.get_all()) == 0

    cache.put("foo", {"bar": 1})
    cache.put("bar", {"bar": 2})

    assert len(cache.get_all()) == 2
    assert cache.get("bar") == {"bar": 2}
    assert cache.get("foo") == {"bar": 1}

    cache.put("pop", {"bar": 3})

    assert cache.get("foo") == {"bar": 1}
    assert cache.get("bar") is None
    assert cache.get("pop") == {"bar": 3}

    with pytest.raises(ValueError):
        cache.put_all([("foo", {"bar": 1}), ("bar", {"bar": 2}), ("pop", {"bar": 3})])
