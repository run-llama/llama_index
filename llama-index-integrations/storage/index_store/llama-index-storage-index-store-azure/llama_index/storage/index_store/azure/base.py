from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.azure import AzureKVStore


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
    ) -> None:
        """Init a MongoIndexStore."""
        super().__init__(azure_kvstore, namespace=namespace)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from an Azure connection string.

        Args:
            connection_string (str): Azure connection string
            namespace (Optional[str]): namespace for the AzureIndexStore
        """
        azure_kvstore = AzureKVStore.from_connection_string(connection_string)
        return cls(azure_kvstore, namespace)

    @classmethod
    def from_account_and_key(
        cls,
        account_name: str,
        account_key: str,
        namespace: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from an account name and key.

        Args:
            account_name (str): Azure Storage Account Name
            account_key (str): Azure Storage Account Key
            namespace (Optional[str]): namespace for the AzureIndexStore
        """
        azure_kvstore = AzureKVStore.from_account_and_key(account_name, account_key)
        return cls(azure_kvstore, namespace)

    @classmethod
    def from_sas_token(
        cls,
        endpoint: str,
        sas_token: str,
        namespace: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from a SAS token.

        Args:
            endpoint (str): Azure Table service endpoint
            sas_token (str): Shared Access Signature token
            namespace (Optional[str]): namespace for the AzureIndexStore
        """
        azure_kvstore = AzureKVStore.from_sas_token(endpoint, sas_token)
        return cls(azure_kvstore, namespace)

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        namespace: Optional[str] = None,
    ) -> "AzureIndexStore":
        """Load an AzureIndexStore from an AAD token.

        Args:
            endpoint (str): Azure Table service endpoint
            namespace (Optional[str]): namespace for the AzureIndexStore
        """
        azure_kvstore = AzureKVStore.from_aad_token(endpoint)
        return cls(azure_kvstore, namespace)
