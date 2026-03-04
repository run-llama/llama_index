import os

import pytest
from llama_index.core.base.llms.types import ChatMessage

from llama_index.storage.chat_store.tablestore import TablestoreChatStore


def test_class():
    names_of_base_classes = [b.__name__ for b in TablestoreChatStore.__mro__]
    assert TablestoreChatStore.__name__ in names_of_base_classes


def get_tablestore() -> TablestoreChatStore:
    """Test end to end construction and search."""
    end_point = os.getenv("tablestore_end_point")
    instance_name = os.getenv("tablestore_instance_name")
    access_key_id = os.getenv("tablestore_access_key_id")
    access_key_secret = os.getenv("tablestore_access_key_secret")
    if (
        end_point is None
        or instance_name is None
        or access_key_id is None
        or access_key_secret is None
    ):
        pytest.skip(
            "end_point is None or instance_name is None or "
            "access_key_id is None or access_key_secret is None"
        )

    # 1. create tablestore vector store
    store = TablestoreChatStore(
        endpoint=os.getenv("tablestore_end_point"),
        instance_name=os.getenv("tablestore_instance_name"),
        access_key_id=os.getenv("tablestore_access_key_id"),
        access_key_secret=os.getenv("tablestore_access_key_secret"),
    )
    store.create_table_if_not_exist()
    store.clear_store()
    return store


def test_add_message():
    store = get_tablestore()
    key = "test_add_key"
    message = ChatMessage(content="add_message_test", role="user")
    store.add_message(key, message=message)

    result = store.get_messages(key)

    assert result[0].content == "add_message_test" and result[0].role == "user"


def test_set_and_retrieve_messages():
    store = get_tablestore()
    messages = [
        ChatMessage(content="Tablestore First message", role="user"),
        ChatMessage(content="Tablestore Second message", role="user"),
        ChatMessage(content="Tablestore 第三 message", role="user"),
    ]
    key = "test_set_key"
    store.set_messages(key, messages)

    retrieved_messages = store.get_messages(key)
    assert len(retrieved_messages) == 3
    assert retrieved_messages[0].content == "Tablestore First message"
    assert retrieved_messages[1].content == "Tablestore Second message"
    assert retrieved_messages[2].content == "Tablestore 第三 message"


def test_delete_messages():
    store = get_tablestore()
    messages = [ChatMessage(content="Tablestore Message to delete", role="user")]
    key = "test_delete_key"
    store.set_messages(key, messages)

    deleted_msg = store.delete_messages(key)
    assert len(deleted_msg) == 1
    assert deleted_msg[0].content == "Tablestore Message to delete"

    retrieved_messages = store.get_messages(key)
    assert retrieved_messages == []


def test_delete_specific_message():
    store = get_tablestore()
    messages = [
        ChatMessage(content="Tablestore Keep me", role="user"),
        ChatMessage(content="Tablestore Delete me", role="user"),
    ]
    key = "test_delete_message_key"
    store.set_messages(key, messages)

    store.delete_message(key, 1)
    retrieved_messages = store.get_messages(key)
    assert len(retrieved_messages) == 1
    assert retrieved_messages[0].content == "Tablestore Keep me"


def test_get_keys():
    store = get_tablestore()
    # Add some test data
    store.set_messages("key1", [ChatMessage(content="Test1", role="user")])
    store.set_messages("key2", [ChatMessage(content="Test2", role="user")])

    keys = store.get_keys()
    assert "key1" in keys
    assert "key2" in keys


def test_delete_last_message():
    store = get_tablestore()
    key = "test_delete_last_message"
    messages = [
        ChatMessage(content="Tablestore First message", role="user"),
        ChatMessage(content="Tablestore Last message", role="user"),
    ]
    store.set_messages(key, messages)

    deleted_message = store.delete_last_message(key)

    assert deleted_message.content == "Tablestore Last message"

    remaining_messages = store.get_messages(key)

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "Tablestore First message"


def test_clear_store():
    store = get_tablestore()
    key = "test_clear_store"
    messages = []
    store.set_messages(key, messages)
    assert len(store.get_keys()) > 0
    store.clear_store()
    assert len(store.get_keys()) == 0
