import time
from typing import Dict, Generator, Union

import docker
import pytest
from docker.models.containers import Container
from llama_index.legacy.storage.kvstore.firestore_kvstore import FirestoreKVStore
from llama_index.legacy.storage.kvstore.mongodb_kvstore import MongoDBKVStore
from llama_index.legacy.storage.kvstore.postgres_kvstore import PostgresKVStore
from llama_index.legacy.storage.kvstore.redis_kvstore import RedisKVStore
from llama_index.legacy.storage.kvstore.simple_kvstore import SimpleKVStore

from tests.storage.kvstore.mock_mongodb import MockMongoClient


@pytest.fixture()
def mongo_client() -> MockMongoClient:
    return MockMongoClient()


@pytest.fixture()
def mongo_kvstore(mongo_client: MockMongoClient) -> MongoDBKVStore:
    return MongoDBKVStore(mongo_client=mongo_client)  # type: ignore


@pytest.fixture()
def firestore_kvstore() -> FirestoreKVStore:
    return FirestoreKVStore()


@pytest.fixture()
def simple_kvstore() -> SimpleKVStore:
    return SimpleKVStore()


@pytest.fixture()
def redis_kvstore() -> "RedisKVStore":
    try:
        from redis import Redis

        client = Redis.from_url(url="redis://127.0.0.1:6379")
    except ImportError:
        return RedisKVStore(redis_client=None, redis_url="redis://127.0.0.1:6379")
    return RedisKVStore(redis_client=client)


@pytest.fixture(scope="module")
def postgres_container() -> Generator[Dict[str, Union[str, Container]], None, None]:
    # Define PostgreSQL settings
    postgres_image = "postgres:latest"
    postgres_env = {
        "POSTGRES_DB": "testdb",
        "POSTGRES_USER": "testuser",
        "POSTGRES_PASSWORD": "testpassword",
    }
    postgres_ports = {"5432/tcp": 5432}
    container = None
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

        # Wait for PostgreSQL to start
        time.sleep(10)  # Adjust the sleep time if necessary

        # Return connection information
        yield {
            "container": container,
            "connection_string": f"postgresql://testuser:testpassword@0.0.0.0:5432/testdb",
            "async_connection_string": f"postgresql+asyncpg://testuser:testpassword@0.0.0.0:5432/testdb",
        }
    finally:
        # Stop and remove the container
        if container:
            container.stop()
            container.remove()
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
