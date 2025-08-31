from typing import Dict, Generator, Union
import pytest
from docker.models.containers import Container
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.yugabytedb import YugabyteDBChatStore

try:
    import psycopg2  # noqa
    import sqlalchemy  # noqa
    no_packages = False
except ImportError:
    no_packages = True


def test_class():
    names_of_base_classes = [b.__name__ for b in YugabyteDBChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


@pytest.fixture()
def yugabytedb_connection() -> Generator[Dict[str, Union[str, Container]], None, None]:
    # Define YugabyteDB settings
    YUGABYTEDB_HOST = "localhost"
    YUGABYTEDB_DBNAME= "yugabyte"
    YUGABYTEDB_USER = "yugabyte"
    YUGABYTEDB_PASSWORD = "yugabyte"
    YUGABYTEDB_PORT = 5433
    YUGABYTEDB_LOAD_BALANCE = True
    try:
        connection_string = f"postgresql://{YUGABYTEDB_USER}:{YUGABYTEDB_PASSWORD}@{YUGABYTEDB_HOST}:{YUGABYTEDB_PORT}/{YUGABYTEDB_DBNAME}?load_balance={YUGABYTEDB_LOAD_BALANCE}"
        psycopg2.connect(connection_string)
        yield {
            "connection_string": connection_string,
        }
    except Exception as e:
        print(f"Could not connect to yugabytedb: {e}")


@pytest.fixture()
def yugabytedb_chat_store(
    yugabytedb_connection: Dict[str, Union[str, Container]],
) -> Generator[YugabyteDBChatStore, None, None]:
    if no_packages:
        pytest.skip("psycopg2 or sqlalchemy not installed")
    chat_store = None
    try:
        chat_store = YugabyteDBChatStore.from_uri(
            uri=yugabytedb_connection["connection_string"],
            use_jsonb=True,
        )
        yield chat_store
    finally:
        if chat_store:
            keys = chat_store.get_keys()
            for key in keys:
                chat_store.delete_messages(key)


@pytest.mark.skipif(no_packages, reason="psycopg2 or sqlalchemy not installed")
def test_yugabytedb_add_message(yugabytedb_chat_store: YugabyteDBChatStore):
    key = "test_add_key"

    message = ChatMessage(content="add_message_test", role="user")
    yugabytedb_chat_store.add_message(key, message=message)

    result = yugabytedb_chat_store.get_messages(key)

    assert result[0].content == "add_message_test" and result[0].role == "user"


@pytest.mark.skipif(no_packages, reason="psycopg2 or sqlalchemy not installed")
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


@pytest.mark.skipif(no_packages, reason="psycopg2 or sqlalchemy not installed")
def test_delete_messages(yugabytedb_chat_store: YugabyteDBChatStore):
    messages = [ChatMessage(content="Message to delete", role="user")]
    key = "test_delete_key"
    yugabytedb_chat_store.set_messages(key, messages)

    yugabytedb_chat_store.delete_messages(key)
    retrieved_messages = yugabytedb_chat_store.get_messages(key)
    assert retrieved_messages == []


@pytest.mark.skipif(no_packages, reason="psycopg2 or sqlalchemy not installed")
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


@pytest.mark.skipif(no_packages, reason="psycopg2 or sqlalchemy not installed")
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


@pytest.mark.skipif(no_packages, reason="psycopg2 or sqlalchemy not installed")
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

