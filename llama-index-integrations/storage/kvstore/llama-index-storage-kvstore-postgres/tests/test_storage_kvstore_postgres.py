import pytest
from importlib.util import find_spec
from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.postgres import PostgresKVStore

no_packages = (
    find_spec("psycopg2") is None
    or find_spec("sqlalchemy") is None
    or find_spec("asyncpg") is None
)


def test_class():
    names_of_base_classes = [b.__name__ for b in PostgresKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_initialization():
    errors = []
    try:
        pgstore1 = PostgresKVStore(table_name="mytable")
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore2 = PostgresKVStore(
            table_name="mytable", connection_string="connection_string"
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore3 = PostgresKVStore(
            table_name="mytable", async_connection_string="async_connection_string"
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore4 = PostgresKVStore(
            table_name="mytable",
            connection_string="connection_string",
            async_connection_string="async_connection_string",
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    assert sum(errors) == 3
    assert pgstore4._engine is None
    assert pgstore4._async_engine is None
