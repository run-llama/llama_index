import docker
import time
from typing import Dict, Generator, Union
import pytest
from docker.models.containers import Container
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.yugabytedb import YugabyteDBChatStore


def test_class():
    names_of_base_classes = [b.__name__ for b in YugabyteDBChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


@pytest.fixture()
def yugabytedb_container() -> Generator[Dict[str, Union[str, Container]], None, None]:
    # Define YugabyteDB settings
    YUGABYTEDB_HOST = "0.0.0.0"
    YUGABYTEDB_DBNAME = "yugabyte"
    YUGABYTEDB_USER = "yugabyte"
    YUGABYTEDB_PASSWORD = "yugabyte"
    YUGABYTEDB_PORT = 5433
    YUGABYTEDB_LOAD_BALANCE = False

    yugabytedb_image = "yugabytedb/yugabyte:2.25.2.0-b359"
    yugabytedb_command = ["bin/yugabyted", "start", "--background=false"]

    # Port mapping (host:container)
    yugabytedb_ports = {"5433/tcp": 5433}

    container = None
    try:
        # Initialize Docker client
        client = docker.from_env()

        # Run YugabyteDB container
        container = client.containers.run(
            yugabytedb_image,
            command=yugabytedb_command,
            ports=yugabytedb_ports,
            name="yugabyte",
            detach=True,
        )

        # Reload to fetch latest state
        container.reload()

        print(f"Container started with ID: {container.id}")

        # Wait for PostgreSQL to start
        time.sleep(10)  # Adjust the sleep time if necessary

        connection_string = f"yugabytedb+psycopg2://{YUGABYTEDB_USER}:{YUGABYTEDB_PASSWORD}@{YUGABYTEDB_HOST}:{YUGABYTEDB_PORT}/{YUGABYTEDB_DBNAME}?load_balance={YUGABYTEDB_LOAD_BALANCE}"
        yield {
            "container": container,
            "connection_string": connection_string,
        }
    except Exception as e:
        print(f"Error: {e!s}")
    finally:
        # Stop and remove the container
        if container:
            container.stop()
            container.remove()
            client.close()


@pytest.fixture()
def yugabytedb_chat_store(
    yugabytedb_container: Dict[str, Union[str, Container]],
) -> Generator[YugabyteDBChatStore, None, None]:
    chat_store = None
    try:
        chat_store = YugabyteDBChatStore.from_uri(
            uri=yugabytedb_container["connection_string"],
            use_jsonb=True,
        )
        yield chat_store
    finally:
        if chat_store:
            keys = chat_store.get_keys()
            for key in keys:
                chat_store.delete_messages(key)


def test_yugabytedb_add_message(yugabytedb_chat_store: YugabyteDBChatStore):
    key = "test_add_key"

    message = ChatMessage(content="add_message_test", role="user")
    yugabytedb_chat_store.add_message(key, message=message)

    result = yugabytedb_chat_store.get_messages(key)

    assert result[0].content == "add_message_test" and result[0].role == "user"


def test_set_and_retrieve_messages(yugabytedb_chat_store: YugabyteDBChatStore):
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Second message", role="user"),
    ]
    key = "test_set_key"
    yugabytedb_chat_store.set_messages(key, messages)

    retrieved_messages = yugabytedb_chat_store.get_messages(key)
    assert len(retrieved_messages) == 2
    assert retrieved_messages[0].content == "First message"
    assert retrieved_messages[1].content == "Second message"


def test_delete_messages(yugabytedb_chat_store: YugabyteDBChatStore):
    messages = [ChatMessage(content="Message to delete", role="user")]
    key = "test_delete_key"
    yugabytedb_chat_store.set_messages(key, messages)

    yugabytedb_chat_store.delete_messages(key)
    retrieved_messages = yugabytedb_chat_store.get_messages(key)
    assert retrieved_messages == []


def test_delete_specific_message(yugabytedb_chat_store: YugabyteDBChatStore):
    messages = [
        ChatMessage(content="Keep me", role="user"),
        ChatMessage(content="Delete me", role="user"),
    ]
    key = "test_delete_message_key"
    yugabytedb_chat_store.set_messages(key, messages)

    yugabytedb_chat_store.delete_message(key, 1)
    retrieved_messages = yugabytedb_chat_store.get_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Keep me"


def test_get_keys(yugabytedb_chat_store: YugabyteDBChatStore):
    # Add some test data
    yugabytedb_chat_store.set_messages(
        "key1", [ChatMessage(content="Test1", role="user")]
    )
    yugabytedb_chat_store.set_messages(
        "key2", [ChatMessage(content="Test2", role="user")]
    )

    keys = yugabytedb_chat_store.get_keys()
    assert "key1" in keys
    assert "key2" in keys


def test_delete_last_message(yugabytedb_chat_store: YugabyteDBChatStore):
    key = "test_delete_last_message"
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Last message", role="user"),
    ]
    yugabytedb_chat_store.set_messages(key, messages)

    deleted_message = yugabytedb_chat_store.delete_last_message(key)

    assert deleted_message.content == "Last message"

    remaining_messages = yugabytedb_chat_store.get_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "First message"
