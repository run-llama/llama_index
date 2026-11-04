import os
import uuid
from typing import Generator

import pytest
from llama_index.core.llms import ChatMessage
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.azurecosmosmongovcore import (
    AzureCosmosMongoVCoreChatStore,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in AzureCosmosMongoVCoreChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")


@pytest.fixture()
def chat_store() -> Generator[AzureCosmosMongoVCoreChatStore, None, None]:
    """Chat store backed by a MongoDB-compatible server, skipped if none runs."""
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except PyMongoError:
        client.close()
        pytest.skip(f"No MongoDB server reachable at {MONGODB_URI}")

    db_name = f"test_db_{uuid.uuid4().hex[:8]}"
    store = AzureCosmosMongoVCoreChatStore(
        mongo_client=client, db_name=db_name, collection_name="chat_store"
    )
    try:
        yield store
    finally:
        client.drop_database(db_name)
        client.close()


def test_set_and_get_messages(chat_store: AzureCosmosMongoVCoreChatStore):
    messages = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi"),
    ]
    chat_store.set_messages("key", messages)
    assert chat_store.get_messages("key") == messages

    # set_messages replaces the stored messages
    chat_store.set_messages("key", messages[:1])
    assert chat_store.get_messages("key") == messages[:1]


def test_add_message(chat_store: AzureCosmosMongoVCoreChatStore):
    first = ChatMessage(role="user", content="first")
    second = ChatMessage(role="assistant", content="second")

    # works for a new key and appends to an existing one
    chat_store.add_message("key", first)
    chat_store.add_message("key", second)

    assert chat_store.get_messages("key") == [first, second]
    assert chat_store._collection.count_documents({}) == 1


def test_get_keys(chat_store: AzureCosmosMongoVCoreChatStore):
    chat_store.set_messages("key1", [ChatMessage(role="user", content="a")])
    chat_store.add_message("key2", ChatMessage(role="user", content="b"))

    assert sorted(chat_store.get_keys()) == ["key1", "key2"]


def test_delete_messages(chat_store: AzureCosmosMongoVCoreChatStore):
    messages = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="user", content="b"),
        ChatMessage(role="user", content="c"),
    ]
    chat_store.set_messages("key", messages)

    assert chat_store.delete_message("key", 1) == messages[1]
    assert chat_store.delete_last_message("key") == messages[2]
    assert chat_store.get_messages("key") == messages[:1]

    assert chat_store.delete_messages("key") == messages[:1]
    assert chat_store.get_messages("key") == []
    assert chat_store.get_keys() == []
