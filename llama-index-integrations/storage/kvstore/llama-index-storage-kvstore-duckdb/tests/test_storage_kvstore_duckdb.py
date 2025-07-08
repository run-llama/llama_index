from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from llama_index.vector_stores.duckdb.base import DuckDBVectorStore
import pytest


def test_class():
    names_of_base_classes = [b.__name__ for b in DuckDBKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


def test_init():
    kv_store = DuckDBKVStore()
    assert kv_store.database_name == ":memory:"
    assert kv_store.table_name == "keyvalue"
    assert kv_store.persist_dir == "./storage"
    assert kv_store.client is not None
    assert kv_store.table is not None


def test_from_vector_store():
    vector_store = DuckDBVectorStore()
    kv_store = DuckDBKVStore.from_vector_store(duckdb_vector_store=vector_store)
    assert kv_store.database_name == vector_store.database_name
    assert kv_store.table_name == "keyvalue"
    assert kv_store.persist_dir == vector_store.persist_dir
    assert kv_store.client is not None

    kv_store.put("id_1", {"name": "John Doe", "text": "Hello, world!"})

    results = kv_store.get_all()
    assert results["id_1"] == {"name": "John Doe", "text": "Hello, world!"}


@pytest.fixture
def kv_store():
    return DuckDBKVStore()


def test_put(kv_store: DuckDBKVStore):
    key = "id_1"
    value = {"name": "John Doe", "text": "Hello, world!"}

    _ = kv_store.put(key, value)


def test_put_twice(kv_store: DuckDBKVStore):
    key = "id_1"
    value = {"name": "John Doe", "text": "Hello, world!"}
    value_updated = {"name": "Jane Doe", "text": "Hello, world!"}

    _ = kv_store.put(key, value)

    _ = kv_store.put(key, value_updated)

    assert kv_store.get(key) == value_updated


def test_put_get(kv_store: DuckDBKVStore):
    key = "id_1"
    value = {"name": "John Doe", "text": "Hello, world!"}

    _ = kv_store.put(key, value)

    assert kv_store.get(key) == value


def test_put_get_collection(kv_store: DuckDBKVStore):
    key = "id_1"
    value = {"name": "John Doe", "text": "Hello, world!"}

    _ = kv_store.put(key, value, collection="collection_1")

    assert kv_store.get(key, collection="collection_1") == value


def test_put_get_all(kv_store: DuckDBKVStore):
    key_1 = "id_1"
    value_1 = {"name": "John Doe", "text": "Hello, world!"}

    key_2 = "id_2"
    value_2 = {"name": "Jane Doe", "text": "Hello, world!"}

    _ = kv_store.put(key_1, value_1)
    _ = kv_store.put(key_2, value_2)

    results = kv_store.get_all()

    assert results[key_1] == value_1
    assert results[key_2] == value_2


def test_delete(kv_store: DuckDBKVStore):
    key = "id_1"
    value = {"name": "John Doe", "text": "Hello, world!"}

    _ = kv_store.put(key, value)

    assert kv_store.get(key) == value

    _ = kv_store.delete(key)

    assert kv_store.get(key) is None


def test_delete_collection(kv_store: DuckDBKVStore):
    key = "id_1"
    value = {"name": "John Doe", "text": "Hello, world!"}

    _ = kv_store.put(key, value, collection="collection_1")

    assert kv_store.get(key, collection="collection_1") == value

    _ = kv_store.delete(key, collection="collection_1")

    assert kv_store.get(key, collection="collection_1") is None
