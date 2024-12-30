from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.azure import AzureKVStore
from llama_index.storage.kvstore.azure.base import ServiceMode


class AzureIndexStore(KVIndexStore):
    """Azure Table Index store.

    Args:
        azure_kvstore (AzureKVStore): Azure key-value store
        namespace (str): namespace for the index store
    """

    def __init__(
        self,
        azure_kvstore: AzureKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a MongoIndexStore."""
        super().__init__(azure_kvstore, namespace, collection_suffix)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from an Azure connection string.

        Args:
            connection_string (str): Azure connection string
            namespace (Optional[str]): namespace for the AzureIndexStore
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_connection_string(
            connection_string, service_mode, partition_key
        )
        return cls(azure_kvstore, namespace, collection_suffix)

    @classmethod
    def from_account_and_key(
        cls,
        account_name: str,
        account_key: str,
        namespace: Optional[str] = None,
        endpoint: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from an account name and key.

        Args:
            account_name (str): Azure Storage Account Name
            account_key (str): Azure Storage Account Key
            namespace (Optional[str]): namespace for the AzureIndexStore
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_account_and_key(
            account_name, account_key, endpoint, service_mode, partition_key
        )
        return cls(azure_kvstore, namespace, collection_suffix)

    @classmethod
    def from_account_and_id(
        cls,
        account_name: str,
        namespace: Optional[str] = None,
        endpoint: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from an account name and managed ID.

        Args:
            account_name (str): Azure Storage Account Name
            namespace (Optional[str]): namespace for the AzureIndexStore
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_account_and_id(
            account_name, endpoint, service_mode, partition_key
        )
        return cls(azure_kvstore, namespace, collection_suffix)

    @classmethod
    def from_sas_token(
        cls,
        endpoint: str,
        sas_token: str,
        namespace: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from a SAS token.

        Args:
            endpoint (str): Azure Table service endpoint
            sas_token (str): Shared Access Signature token
            namespace (Optional[str]): namespace for the AzureIndexStore
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_sas_token(
            endpoint, sas_token, service_mode, partition_key
        )
        return cls(azure_kvstore, namespace, collection_suffix)

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        namespace: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from an AAD token.

        Args:
            endpoint (str): Azure Table service endpoint
            namespace (Optional[str]): namespace for the AzureIndexStore
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_aad_token(
            endpoint, service_mode, partition_key
        )
        return cls(azure_kvstore, namespace, collection_suffix)
