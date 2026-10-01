from unittest.mock import MagicMock

import pytest
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.azurecosmosmongovcore import (
    AzureCosmosMongoVCoreChatStore,
)
from llama_index.storage.chat_store.azuredocumentdb import AzureDocumentDBChatStore


def test_class():
    names_of_base_classes = [b.__name__ for b in AzureCosmosMongoVCoreChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes
    assert issubclass(AzureCosmosMongoVCoreChatStore, AzureDocumentDBChatStore)


def test_deprecation_warning_and_class_name() -> None:
    with pytest.warns(DeprecationWarning, match="AzureDocumentDBChatStore"):
        chat_store = AzureCosmosMongoVCoreChatStore(
            mongo_client=MagicMock(),
            db_name="test_db",
            collection_name="test_collection",
        )

    assert chat_store.class_name() == "AzureCosmosMongoVCoreChatStore"
