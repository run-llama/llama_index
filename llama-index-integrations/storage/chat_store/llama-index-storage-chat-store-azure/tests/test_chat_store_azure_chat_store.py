import pytest
from unittest.mock import AsyncMock, MagicMock
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.azure import AzureChatStore
from llama_index.core.llms import ChatMessage
from azure.data.tables import TableServiceClient, TableClient, TableEntity
from azure.data.tables.aio import TableServiceClient as AsyncTableServiceClient


def test_class():
    names_of_base_classes = [b.__name__ for b in AzureChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


@pytest.fixture()
def azure_chat_store():
    mock_table_service_client = MagicMock(spec=TableServiceClient)
    mock_atable_service_client = AsyncMock(spec=AsyncTableServiceClient)
    return AzureChatStore(mock_table_service_client, mock_atable_service_client)


def test_set_messages(azure_chat_store):
    key = "test_key"
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]

    mock_chat_client = AsyncMock(spec=TableClient)
    mock_metadata_client = AsyncMock(spec=TableClient)

    azure_chat_store._atable_service_client.create_table_if_not_exists.side_effect = [
        mock_chat_client,
        mock_metadata_client,
    ]

    # Mock the query_entities method to return an empty list
    mock_chat_client.query_entities.return_value = AsyncMock()
    mock_chat_client.query_entities.return_value.__aiter__.return_value = []
    mock_chat_client.submit_transaction = AsyncMock()
    mock_metadata_client.upsert_entity = AsyncMock()

    azure_chat_store.set_messages(key, messages)

    azure_chat_store._atable_service_client.create_table_if_not_exists.assert_any_call(
        azure_chat_store.chat_table_name
    )
    azure_chat_store._atable_service_client.create_table_if_not_exists.assert_any_call(
        azure_chat_store.metadata_table_name
    )
    mock_chat_client.submit_transaction.assert_called_once()
    mock_metadata_client.upsert_entity.assert_called_once()


def test_get_messages(azure_chat_store):
    key = "test_key"

    mock_chat_client = AsyncMock(spec=TableClient)
    azure_chat_store._atable_service_client.create_table_if_not_exists.return_value = (
        mock_chat_client
    )

    # Create mock TableEntity objects
    mock_entities = [
        TableEntity(
            PartitionKey=key, RowKey="0000000000", role="user", content="Hello"
        ),
        TableEntity(
            PartitionKey=key, RowKey="0000000001", role="assistant", content="Hi there!"
        ),
    ]

    # Set up the mock to return an async iterator of the mock entities
    mock_chat_client.query_entities.return_value = AsyncMock()
    mock_chat_client.query_entities.return_value.__aiter__.return_value = mock_entities

    result = azure_chat_store.get_messages(key)

    azure_chat_store._atable_service_client.create_table_if_not_exists.assert_called_once_with(
        azure_chat_store.chat_table_name
    )
    mock_chat_client.query_entities.assert_called_once_with(f"PartitionKey eq '{key}'")
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].content == "Hello"
    assert result[1].role == "assistant"
    assert result[1].content == "Hi there!"
