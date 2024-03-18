import pytest
from llama_index.legacy.data_structs.data_structs import IndexGraph
from llama_index.legacy.storage.index_store.postgres_index_store import (
    PostgresIndexStore,
)
from llama_index.legacy.storage.kvstore.postgres_kvstore import PostgresKVStore

try:
    import asyncpg  # noqa
    import psycopg2  # noqa
    import sqlalchemy  # noqa

    no_packages = False
except ImportError:
    no_packages = True


@pytest.fixture()
def postgres_indexstore(postgres_kvstore: PostgresKVStore) -> PostgresIndexStore:
    return PostgresIndexStore(postgres_kvstore=postgres_kvstore)


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_postgres_index_store(postgres_indexstore: PostgresIndexStore) -> None:
    index_struct = IndexGraph()
    index_store = postgres_indexstore

    index_store.add_index_struct(index_struct)
    assert index_store.get_index_struct(struct_id=index_struct.index_id) == index_struct
