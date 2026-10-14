import warnings
from typing import Optional

from llama_index.storage.chat_store.azuredocumentdb import AzureDocumentDBChatStore
from pymongo import MongoClient


class AzureCosmosMongoVCoreChatStore(AzureDocumentDBChatStore):
    """Deprecated compatibility wrapper for AzureDocumentDBChatStore."""

    def __init__(
        self,
        mongo_client: MongoClient,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        warnings.warn(
            "AzureCosmosMongoVCoreChatStore is deprecated; use "
            "AzureDocumentDBChatStore from llama_index.storage.chat_store.azuredocumentdb instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            mongo_client=mongo_client,
            uri=uri,
            host=host,
            port=port,
            db_name=db_name,
            collection_name=collection_name,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get the legacy class name."""
        return "AzureCosmosMongoVCoreChatStore"
