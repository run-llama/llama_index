from typing import Any, Dict, Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.azurecosmosnosql import AzureCosmosNoSqlKVStore

DEFAULT_INDEX_DATABASE = "IndexStoreDB"
DEFAULT_INDEX_CONTAINER = "IndexStoreContainer"


class AzureCosmosNoSqlIndexStore(KVIndexStore):
    """Creates an Azure Cosmos DB NoSql Index Store."""

    def __init__(
        self,
        azure_cosmos_nosql_kvstore: AzureCosmosNoSqlKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Initializes the Azure Cosmos NoSql Index Store."""
        super().__init__(azure_cosmos_nosql_kvstore, namespace, collection_suffix)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        index_db_name: str = DEFAULT_INDEX_DATABASE,
        index_container_name: str = DEFAULT_INDEX_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlIndexStore":
        """Creates an instance of Azure Cosmos DB NoSql KV Store using a connection string."""
        azure_cosmos_nosql_kvstore = AzureCosmosNoSqlKVStore.from_connection_string(
            connection_string,
            index_db_name,
            index_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )
        namespace = index_db_name + "." + index_container_name
        return cls(azure_cosmos_nosql_kvstore, namespace)

    @classmethod
    def from_account_and_key(
        cls,
        endpoint: str,
        key: str,
        index_db_name: str = DEFAULT_INDEX_DATABASE,
        index_container_name: str = DEFAULT_INDEX_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlIndexStore":
        """Creates an instance of Azure Cosmos DB NoSql KV Store using an account endpoint and key."""
        azure_cosmos_nosql_kvstore = AzureCosmosNoSqlKVStore.from_account_and_key(
            endpoint,
            key,
            index_db_name,
            index_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )
        namespace = index_db_name + "." + index_container_name
        return cls(azure_cosmos_nosql_kvstore, namespace)

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        index_db_name: str = DEFAULT_INDEX_DATABASE,
        index_container_name: str = DEFAULT_INDEX_CONTAINER,
        cosmos_container_properties: Dict[str, Any] = None,
        cosmos_database_properties: Dict[str, Any] = None,
    ) -> "AzureCosmosNoSqlIndexStore":
        """Creates an instance of Azure Cosmos DB NoSql KV Store using an aad token."""
        azure_cosmos_nosql_kvstore = AzureCosmosNoSqlKVStore.from_aad_token(
            endpoint,
            index_db_name,
            index_container_name,
            cosmos_container_properties,
            cosmos_database_properties,
        )
        namespace = index_db_name + "." + index_container_name
        return cls(azure_cosmos_nosql_kvstore, namespace)
