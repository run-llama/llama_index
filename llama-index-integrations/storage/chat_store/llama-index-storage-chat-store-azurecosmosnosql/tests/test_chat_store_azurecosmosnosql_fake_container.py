"""Tests against an in-memory container that enforces the ``ContainerProxy`` API."""

import copy
from typing import Any, Dict
from unittest.mock import MagicMock, create_autospec

import pytest
from azure.cosmos import ContainerProxy, PartitionKey
from azure.cosmos.exceptions import (
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)
from llama_index.core.llms import ChatMessage
from llama_index.storage.chat_store.azurecosmosnosql import AzureCosmosNoSqlChatStore


def _fake_container() -> MagicMock:
    """Autospec'd container (so wrong call signatures raise) backed by a dict."""
    items: Dict[str, Dict[str, Any]] = {}
    container = create_autospec(ContainerProxy, instance=True)

    def create_item(body, *args, **kwargs):
        if body["id"] in items:
            raise CosmosResourceExistsError(message="Conflict")
        items[body["id"]] = copy.deepcopy(body)
        return copy.deepcopy(body)

    def upsert_item(body, *args, **kwargs):
        items[body["id"]] = copy.deepcopy(body)
        return copy.deepcopy(body)

    def read_item(item, partition_key, *args, **kwargs):
        if item not in items or partition_key != item:
            raise CosmosResourceNotFoundError(message="Not found")
        return copy.deepcopy(items[item])

    def delete_item(item, partition_key, *args, **kwargs):
        if item not in items or partition_key != item:
            raise CosmosResourceNotFoundError(message="Not found")
        del items[item]

    def read_all_items(*args, **kwargs):
        return [copy.deepcopy(v) for v in items.values()]

    container.create_item.side_effect = create_item
    container.upsert_item.side_effect = upsert_item
    container.read_item.side_effect = read_item
    container.delete_item.side_effect = delete_item
    container.read_all_items.side_effect = read_all_items
    return container


@pytest.fixture()
def store() -> AzureCosmosNoSqlChatStore:
    client = MagicMock()
    database = client.create_database_if_not_exists.return_value
    database.create_container_if_not_exists.return_value = _fake_container()
    return AzureCosmosNoSqlChatStore(
        cosmos_client=client,
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        cosmos_database_properties={},
    )


def _contents(store: AzureCosmosNoSqlChatStore, key: str) -> list:
    return [m.content for m in store.get_messages(key)]


def test_get_messages_missing_key_returns_empty(store: AzureCosmosNoSqlChatStore):
    assert store.get_messages("missing") == []


def test_set_and_get_messages(store: AzureCosmosNoSqlChatStore):
    store.set_messages("k", [ChatMessage(role="user", content="a")])
    assert _contents(store, "k") == ["a"]


def test_add_message_appends(store: AzureCosmosNoSqlChatStore):
    store.add_message("k", ChatMessage(role="user", content="a"))
    store.add_message("k", ChatMessage(role="assistant", content="b"))
    assert _contents(store, "k") == ["a", "b"]


def test_delete_message_and_last_message(store: AzureCosmosNoSqlChatStore):
    store.set_messages(
        "k", [ChatMessage(role="user", content=c) for c in ("a", "b", "c")]
    )

    deleted = store.delete_message("k", 1)
    assert deleted is not None and deleted.content == "b"
    assert _contents(store, "k") == ["a", "c"]

    last = store.delete_last_message("k")
    assert last is not None and last.content == "c"
    assert _contents(store, "k") == ["a"]

    assert store.delete_message("k", 5) is None


def test_delete_messages(store: AzureCosmosNoSqlChatStore):
    store.set_messages("k", [ChatMessage(role="user", content="a")])

    deleted = store.delete_messages("k")
    assert [m.content for m in deleted] == ["a"]
    assert store.get_messages("k") == []
    assert store.get_keys() == []

    # Deleting a key that does not exist is not an error.
    assert store.delete_messages("k") == []
