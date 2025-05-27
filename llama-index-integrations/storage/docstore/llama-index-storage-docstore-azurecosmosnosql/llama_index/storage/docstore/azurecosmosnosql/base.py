from typing import Any, Dict, Optional

from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.azurecosmosnosql import AzureCosmosNoSqlKVStore

DEFAULT_DOCUMENT_DATABASE = "DocumentStoreDB"
DEFAULT_DOCUMENT_CONTAINER = "DocumentStoreContainer"


class AzureCosmosNoSqlDocumentStore(BaseKVStore):
    """Creates an AzureCosmosNoSqlDocumentStore."""

    def __init__(
        self,
        azure_cosmos_nosql_kvstore: AzureCosmosNoSqlKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Initializes the AzureCosmosNoSqlDocumentStore."""
        super().__init__(azure_cosmos_nosql_kvstore, namespace, collection_suffix)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        document_db_name: str = DEFAULT_DOCUMENT_DATABASE,
        document_container_name: str = DEFAULT_DOCUMENT_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlDocumentStore":
        """Creates an instance of AzureCosmosNoSqlDocumentStore using a connection string."""
        azure_cosmos_nosql_kvstore = AzureCosmosNoSqlKVStore.from_connection_string(
            connection_string,
            document_db_name,
            document_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )
        namespace = document_db_name + "." + document_container_name
        return cls(azure_cosmos_nosql_kvstore, namespace)

    @classmethod
    def from_account_and_key(
        cls,
        endpoint: str,
        key: str,
        document_db_name: str = DEFAULT_DOCUMENT_DATABASE,
        document_container_name: str = DEFAULT_DOCUMENT_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlDocumentStore":
        """Creates an instance of AzureCosmosNoSqlDocumentStore using an account endpoint and key."""
        azure_cosmos_nosql_kvstore = AzureCosmosNoSqlKVStore.from_account_and_key(
            endpoint,
            key,
            document_db_name,
            document_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )
        namespace = document_db_name + "." + document_container_name
        return cls(azure_cosmos_nosql_kvstore, namespace)

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        document_db_name: str = DEFAULT_DOCUMENT_DATABASE,
        document_container_name: str = DEFAULT_DOCUMENT_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlDocumentStore":
        """Creates an instance of AzureCosmosNoSqlDocumentStore using an aad token."""
        azure_cosmos_nosql_kvstore = AzureCosmosNoSqlKVStore.from_aad_token(
            endpoint,
            document_db_name,
            document_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )
        namespace = document_db_name + "." + document_container_name
        return cls(azure_cosmos_nosql_kvstore, namespace)
