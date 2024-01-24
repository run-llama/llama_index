import pytest
from llama_index.data_structs.data_structs import IndexGraph
from llama_index.storage.index_store.postgres_index_store import PostgresIndexStore
from llama_index.storage.kvstore.postgres_kvstore import PostgresKVStore

try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None  # type: ignore


@pytest.fixture()
def postgres_indexstore(postgres_kvstore: PostgresKVStore) -> PostgresIndexStore:
    return PostgresIndexStore(postgres_kvstore=postgres_kvstore)


@pytest.mark.skipif(sqlalchemy is None, reason="sqlalchemy not installed")
def test_postgres_index_store(postgres_indexstore: PostgresIndexStore) -> None:
    index_struct = IndexGraph()
    index_store = postgres_indexstore

    index_store.add_index_struct(index_struct)
    assert index_store.get_index_struct(struct_id=index_struct.index_id) == index_struct
