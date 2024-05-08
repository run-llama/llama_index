from typing import Optional

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE
from llama_index.storage.kvstore.azure import AzureKVStore


class AzureDocumentStore(KVDocumentStore):
    """Azure Document (Node) store.

    An Azure Table store for Document and Node objects.

    Args:
        azure_kvstore (AzureKVStore): Azure key-value store
        namespace (Optional[str]): namespace for the AzureDocumentStore
        node_collection_suffix (Optional[str]): suffix for node collection
        ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
        metadata_collection_suffix (Optional[str]): suffix for metadata collection
        batch_size (int): batch size for batch operations
    """

    def __init__(
        self,
        azure_kvstore: AzureKVStore,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Init an AzureDocumentStore."""
        super().__init__(
            azure_kvstore,
            namespace=namespace,
            batch_size=batch_size,
            node_collection_suffix=node_collection_suffix,
            ref_doc_collection_suffix=ref_doc_collection_suffix,
            metadata_collection_suffix=metadata_collection_suffix,
        )

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
    ) -> "AzureDocumentStore":
        """Load an AzureDocumentStore from an Azure connection string.

        Args:
            connection_string (str): Azure connection string
            namespace (Optional[str]): namespace for the AzureDocumentStore
            node_collection_suffix (Optional[str]): suffix for node collection
            ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
            metadata_collection_suffix (Optional[str]): suffix for metadata collection
        """
        azure_kvstore = AzureKVStore.from_connection_string(connection_string)
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
        )

    @classmethod
    def from_account_and_key(
        cls,
        account_name: str,
        account_key: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
    ) -> "AzureDocumentStore":
        """Load an AzureDocumentStore from an account name and key.

        Args:
            account_name (str): Azure Storage Account Name
            account_key (str): Azure Storage Account Key
            namespace (Optional[str]): namespace for the AzureDocumentStore
            node_collection_suffix (Optional[str]): suffix for node collection
            ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
            metadata_collection_suffix (Optional[str]): suffix for metadata collection
        """
        azure_kvstore = AzureKVStore.from_account_and_key(account_name, account_key)
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
        )

    @classmethod
    def from_sas_token(
        cls,
        endpoint: str,
        sas_token: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
    ) -> "AzureDocumentStore":
        """Load an AzureDocumentStore from a SAS token.

        Args:
            endpoint (str): Azure Table service endpoint
            sas_token (str): Shared Access Signature token
            namespace (Optional[str]): namespace for the AzureDocumentStore
            node_collection_suffix (Optional[str]): suffix for node collection
            ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
            metadata_collection_suffix (Optional[str]): suffix for metadata collection
        """
        azure_kvstore = AzureKVStore.from_sas_token(endpoint, sas_token)
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
        )

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
    ) -> "AzureDocumentStore":
        """Load an AzureDocumentStore from an AAD token.

        Args:
            endpoint (str): Azure Table service endpoint
            namespace (Optional[str]): namespace for the AzureDocumentStore
            node_collection_suffix (Optional[str]): suffix for node collection
            ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
            metadata_collection_suffix (Optional[str]): suffix for metadata collection
        """
        azure_kvstore = AzureKVStore.from_aad_token(endpoint)
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
        )
