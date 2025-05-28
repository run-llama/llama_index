import docker
import pytest
import time
from docker.models.containers import Container
from importlib.util import find_spec
from typing import Dict, Generator, Union

from llama_index.storage.kvstore.postgres import PostgresKVStore

no_packages = (
    find_spec("psycopg2") is None
    or find_spec("sqlalchemy") is None
    or find_spec("asyncpg") is None
)


@pytest.fixture()
def postgres_container() -> Generator[Dict[str, Union[str, Container]], None, None]:
    postgres_image = "postgres:latest"
    postgres_env = {
        "POSTGRES_DB": "testdb",
        "POSTGRES_USER": "testuser",
        "POSTGRES_PASSWORD": "testpassword",
    }
    # Let Docker choose available port
    postgres_ports = {"5432/tcp": None}
    container = None
    client = None

    try:
        # Initialize Docker client
        client = docker.from_env()

        # Run PostgreSQL container
        container = client.containers.run(
            postgres_image, environment=postgres_env, ports=postgres_ports, detach=True
        )

        # Retrieve the container's port
        container.reload()
        postgres_port = container.attrs["NetworkSettings"]["Ports"]["5432/tcp"][0][
            "HostPort"
        ]

        # Wait for PostgreSQL to be ready by polling
        connection_string = (
            f"postgresql://testuser:testpassword@0.0.0.0:{postgres_port}/testdb"
        )
        async_connection_string = (
            f"postgresql+asyncpg://testuser:testpassword@0.0.0.0:{postgres_port}/testdb"
        )

        # Wait for PostgreSQL to be ready
        max_retries = 30
        retry_interval = 1
        import sqlalchemy

        for i in range(max_retries):
            try:
                engine = sqlalchemy.create_engine(connection_string)
                conn = engine.connect()
                conn.close()
                engine.dispose()
                break
            except Exception as e:
                if i == max_retries - 1:
                    raise Exception(
                        f"Failed to connect to PostgreSQL after {max_retries} attempts: {e}"
                    )
                time.sleep(retry_interval)

        # Return connection information
        yield {
            "container": container,
            "connection_string": connection_string,
            "async_connection_string": async_connection_string,
        }
    finally:
        # Stop and remove the container
        if container:
            try:
                container.stop(timeout=5)
                container.remove()
            except Exception:
                pass  # Ignore errors during cleanup
        if client:
            client.close()


@pytest.fixture()
def postgres_kvstore(
    postgres_container: Dict[str, Union[str, Container]],
) -> Generator[PostgresKVStore, None, None]:
    kvstore = None
    try:
        kvstore = PostgresKVStore(
            connection_string=postgres_container["connection_string"],
            async_connection_string=postgres_container["async_connection_string"],
            table_name="test_kvstore",
            schema_name="test_schema",
            use_jsonb=True,
        )
        yield kvstore
    finally:
        if kvstore:
            keys = kvstore.get_all().keys()
            for key in keys:
                kvstore.delete(key)


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
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
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_from_uri(postgres_container: Dict[str, Union[str, Container]]) -> None:
    kvstore = PostgresKVStore.from_uri(uri=postgres_container["connection_string"])
    output = kvstore.get_all()
    assert len(list(output.keys())) == 0


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio
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
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
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
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio
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
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
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
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio
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
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio
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
