from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.azurecosmosmongovcore import (
    AzureCosmosMongoVCoreKVStore,
)


APP_NAME = "Llama-Index-CDBMongoVCore-IndexStore-Python"


class AzureCosmosMongoVCoreIndexStore(KVIndexStore):
    """Creates an AzureCosmosMongoVCoreIndexStore."""

    def __init__(
        self,
        azure_cosmos_mongo_vcore_kvstore: AzureCosmosMongoVCoreKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Initializes the Azure Cosmos Mongo vCore Index Store."""
        super().__init__(
            azure_cosmos_mongo_vcore_kvstore,
            namespace=namespace,
            collection_suffix=collection_suffix,
        )

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "AzureCosmosMongoVCoreIndexStore":
        """Creates an instance of AzureCosmosMongoVCoreIndexStore using a connection string."""
        azure_cosmos_mongo_vcore_kvstore = (
            AzureCosmosMongoVCoreKVStore.from_connection_string(
                connection_string, db_name=db_name, collection_name=collection_name
            )
        )
        namespace = db_name + "." + collection_name
        return cls(azure_cosmos_mongo_vcore_kvstore, namespace)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "AzureCosmosMongoVCoreIndexStore":
        """Initializes AzureCosmosMongoVCoreIndexStore from an endpoint url and key."""
        azure_cosmos_mongo_vcore_kvstore = (
            AzureCosmosMongoVCoreKVStore.from_host_and_port(
                host, port, db_name=db_name, collection_name=collection_name
            )
        )
        namespace = db_name + "." + collection_name
        return cls(
            azure_cosmos_mongo_vcore_kvstore,
            namespace,
        )
