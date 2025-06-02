import time
from typing import Dict, Generator, Union
import pytest
import docker
from docker.models.containers import Container
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.postgres import PostgresChatStore

try:
    import asyncpg  # noqa
    import psycopg  # noqa
    import sqlalchemy  # noqa

    no_packages = False
except ImportError:
    no_packages = True


def test_class():
    names_of_base_classes = [b.__name__ for b in PostgresChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


@pytest.fixture()
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
@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
def postgres_chat_store(
    postgres_container: Dict[str, Union[str, Container]],
) -> Generator[PostgresChatStore, None, None]:
    chat_store = None
    try:
        chat_store = PostgresChatStore.from_uri(
            uri=postgres_container["connection_string"],
            use_jsonb=True,
        )
        yield chat_store
    finally:
        if chat_store:
            keys = chat_store.get_keys()
            for key in keys:
                chat_store.delete_messages(key)


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
def test_postgres_add_message(postgres_chat_store: PostgresChatStore):
    key = "test_add_key"

    message = ChatMessage(content="add_message_test", role="user")
    postgres_chat_store.add_message(key, message=message)

    result = postgres_chat_store.get_messages(key)

    assert result[0].content == "add_message_test" and result[0].role == "user"


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
def test_set_and_retrieve_messages(postgres_chat_store: PostgresChatStore):
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Second message", role="user"),
    ]
    key = "test_set_key"
    postgres_chat_store.set_messages(key, messages)

    retrieved_messages = postgres_chat_store.get_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First message"
    assert retrieved_messages[1].content == "Second message"


def test_delete_messages(postgres_chat_store: PostgresChatStore):
    messages = [ChatMessage(content="Message to delete", role="user")]
    key = "test_delete_key"
    postgres_chat_store.set_messages(key, messages)

    postgres_chat_store.delete_messages(key)
    retrieved_messages = postgres_chat_store.get_messages(key)
    assert retrieved_messages == []


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
def test_delete_specific_message(postgres_chat_store: PostgresChatStore):
    messages = [
        ChatMessage(content="Keep me", role="user"),
        ChatMessage(content="Delete me", role="user"),
    ]
    key = "test_delete_message_key"
    postgres_chat_store.set_messages(key, messages)

    postgres_chat_store.delete_message(key, 1)
    retrieved_messages = postgres_chat_store.get_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Keep me"


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
async def test_get_keys(postgres_chat_store: PostgresChatStore):
    # Add some test data
    postgres_chat_store.set_messages(
        "key1", [ChatMessage(content="Test1", role="user")]
    )
    postgres_chat_store.set_messages(
        "key2", [ChatMessage(content="Test2", role="user")]
    )

    keys = postgres_chat_store.get_keys()
    assert "key1" in keys
    assert "key2" in keys


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
async def test_delete_last_message(postgres_chat_store: PostgresChatStore):
    key = "test_delete_last_message"
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Last message", role="user"),
    ]
    postgres_chat_store.set_messages(key, messages)

    deleted_message = postgres_chat_store.delete_last_message(key)

    assert deleted_message.content == "Last message"

    remaining_messages = postgres_chat_store.get_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First message"


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
@pytest.mark.asyncio
async def test_async_postgres_add_message(postgres_chat_store: PostgresChatStore):
    key = "test_async_add_key"

    message = ChatMessage(content="async_add_message_test", role="user")
    await postgres_chat_store.async_add_message(key, message=message)

    result = await postgres_chat_store.aget_messages(key)

    assert result[0].content == "async_add_message_test" and result[0].role == "user"


@pytest.mark.asyncio
async def test_async_set_and_retrieve_messages(postgres_chat_store: PostgresChatStore):
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Second async message", role="user"),
    ]
    key = "test_async_set_key"
    await postgres_chat_store.aset_messages(key, messages)

    retrieved_messages = await postgres_chat_store.aget_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First async message"
    assert retrieved_messages[1].content == "Second async message"


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
@pytest.mark.asyncio
async def test_adelete_messages(postgres_chat_store: PostgresChatStore):
    messages = [ChatMessage(content="Async message to delete", role="user")]
    key = "test_async_delete_key"
    await postgres_chat_store.aset_messages(key, messages)

    await postgres_chat_store.adelete_messages(key)
    retrieved_messages = await postgres_chat_store.aget_messages(key)
    assert retrieved_messages == []


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
@pytest.mark.asyncio
async def test_async_delete_specific_message(postgres_chat_store: PostgresChatStore):
    messages = [
        ChatMessage(content="Async keep me", role="user"),
        ChatMessage(content="Async delete me", role="user"),
    ]
    key = "test_adelete_message_key"
    await postgres_chat_store.aset_messages(key, messages)

    await postgres_chat_store.adelete_message(key, 1)
    retrieved_messages = await postgres_chat_store.aget_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Async keep me"


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
@pytest.mark.asyncio
async def test_async_get_keys(postgres_chat_store: PostgresChatStore):
    # Add some test data
    await postgres_chat_store.aset_messages(
        "async_key1", [ChatMessage(content="Test1", role="user")]
    )
    await postgres_chat_store.aset_messages(
        "async_key2", [ChatMessage(content="Test2", role="user")]
    )

    keys = await postgres_chat_store.aget_keys()
    assert "async_key1" in keys
    assert "async_key2" in keys


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
@pytest.mark.asyncio
async def test_async_delete_last_message(postgres_chat_store: PostgresChatStore):
    key = "test_async_delete_last_message"
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Last async message", role="user"),
    ]
    await postgres_chat_store.aset_messages(key, messages)

    deleted_message = await postgres_chat_store.adelete_last_message(key)

    assert deleted_message.content == "Last async message"

    remaining_messages = await postgres_chat_store.aget_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First async message"


@pytest.mark.skipif(no_packages, reason="ayncpg, pscopg and sqlalchemy not installed")
@pytest.mark.asyncio
async def test_async_multimodal_messages(postgres_chat_store: PostgresChatStore):
    key = "test_async_multimodal"
    image_url = "https://images.unsplash.com/photo-1579546929518-9e396f3cc809"

    messages = [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="describe the image."),
                ImageBlock(url=image_url),
            ],
        )
    ]

    await postgres_chat_store.aset_messages(key, messages)

    retrieved_messages = await postgres_chat_store.aget_messages(key)

    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].role == "user"
    assert len(retrieved_messages[0].blocks) == 2
    assert isinstance(retrieved_messages[0].blocks[0], TextBlock)
    assert retrieved_messages[0].blocks[0].text == "describe the image."
    assert isinstance(retrieved_messages[0].blocks[1], ImageBlock)
    assert str(retrieved_messages[0].blocks[1].url) == image_url
