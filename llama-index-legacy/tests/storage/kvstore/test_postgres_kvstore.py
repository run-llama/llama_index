from typing import Dict, Union

import pytest
from docker.models.containers import Container
from llama_index.legacy.storage.kvstore.postgres_kvstore import PostgresKVStore

try:
    import asyncpg  # noqa
    import psycopg2  # noqa
    import sqlalchemy  # noqa

    no_packages = False
except ImportError:
    no_packages = True


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_kvstore_basic(postgres_kvstore: PostgresKVStore) -> None:
    test_key = "test_key_basic"
    test_blob = {"test_obj_key": "test_obj_val"}
    postgres_kvstore.put(test_key, test_blob)
    blob = postgres_kvstore.get(test_key)
    assert blob == test_blob

    blob = postgres_kvstore.get(test_key, collection="non_existent")
    assert blob is None

    deleted = postgres_kvstore.delete(test_key)
    assert deleted


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_from_uri(postgres_container: Dict[str, Union[str, Container]]) -> None:
    kvstore = PostgresKVStore.from_uri(uri=postgres_container["connection_string"])
    output = kvstore.get_all()
    assert len(list(output.keys())) == 0


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio()
async def test_kvstore_async_basic(postgres_kvstore: PostgresKVStore) -> None:
    test_key = "test_key_basic"
    test_blob = {"test_obj_key": "test_obj_val"}
    await postgres_kvstore.aput(test_key, test_blob)
    blob = await postgres_kvstore.aget(test_key)
    assert blob == test_blob

    blob = await postgres_kvstore.aget(test_key, collection="non_existent")
    assert blob is None

    deleted = await postgres_kvstore.adelete(test_key)
    assert deleted


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_kvstore_delete(postgres_kvstore: PostgresKVStore) -> None:
    test_key = "test_key_delete"
    test_blob = {"test_obj_key": "test_obj_val"}
    postgres_kvstore.put(test_key, test_blob)
    blob = postgres_kvstore.get(test_key)
    assert blob == test_blob

    postgres_kvstore.delete(test_key)
    blob = postgres_kvstore.get(test_key)
    assert blob is None


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio()
async def test_kvstore_adelete(postgres_kvstore: PostgresKVStore) -> None:
    test_key = "test_key_delete"
    test_blob = {"test_obj_key": "test_obj_val"}
    await postgres_kvstore.aput(test_key, test_blob)
    blob = await postgres_kvstore.aget(test_key)
    assert blob == test_blob

    await postgres_kvstore.adelete(test_key)
    blob = await postgres_kvstore.aget(test_key)
    assert blob is None


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_kvstore_getall(postgres_kvstore: PostgresKVStore) -> None:
    test_key_1 = "test_key_1"
    test_blob_1 = {"test_obj_key": "test_obj_val"}
    postgres_kvstore.put(test_key_1, test_blob_1)
    blob = postgres_kvstore.get(test_key_1)
    assert blob == test_blob_1
    test_key_2 = "test_key_2"
    test_blob_2 = {"test_obj_key": "test_obj_val"}
    postgres_kvstore.put(test_key_2, test_blob_2)
    blob = postgres_kvstore.get(test_key_2)
    assert blob == test_blob_2

    blob = postgres_kvstore.get_all()
    assert len(blob) == 2

    postgres_kvstore.delete(test_key_1)
    postgres_kvstore.delete(test_key_2)


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio()
async def test_kvstore_agetall(postgres_kvstore: PostgresKVStore) -> None:
    test_key_1 = "test_key_1"
    test_blob_1 = {"test_obj_key": "test_obj_val"}
    await postgres_kvstore.aput(test_key_1, test_blob_1)
    blob = await postgres_kvstore.aget(test_key_1)
    assert blob == test_blob_1
    test_key_2 = "test_key_2"
    test_blob_2 = {"test_obj_key": "test_obj_val"}
    await postgres_kvstore.aput(test_key_2, test_blob_2)
    blob = await postgres_kvstore.aget(test_key_2)
    assert blob == test_blob_2

    blob = await postgres_kvstore.aget_all()
    assert len(blob) == 2

    await postgres_kvstore.adelete(test_key_1)
    await postgres_kvstore.adelete(test_key_2)


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio()
async def test_kvstore_putall(postgres_kvstore: PostgresKVStore) -> None:
    test_key = "test_key_putall_1"
    test_blob = {"test_obj_key": "test_obj_val"}
    test_key2 = "test_key_putall_2"
    test_blob2 = {"test_obj_key2": "test_obj_val2"}
    await postgres_kvstore.aput_all([(test_key, test_blob), (test_key2, test_blob2)])
    blob = await postgres_kvstore.aget(test_key)
    assert blob == test_blob
    blob = await postgres_kvstore.aget(test_key2)
    assert blob == test_blob2

    await postgres_kvstore.adelete(test_key)
    await postgres_kvstore.adelete(test_key2)
