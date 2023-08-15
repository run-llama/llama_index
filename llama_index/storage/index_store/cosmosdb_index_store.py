from typing import Optional

from llama_index.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore


class CosmosDBIndexStore(KVIndexStore):
    """CosmosDB Index store.

    Args:
        mongo_kvstore (MongoDBKVStore): MongoDB key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        mongo_kvstore: MongoDBKVStore,
        namespace: Optional[str] = None,
    ) -> None:
        """Init a CosmosDBIndexStore."""
        super().__init__(mongo_kvstore, namespace=namespace)
        # Replace "/" with "_" in collections, CosmosDB doesn't allow "/"
        self._collection = f"{self._namespace}_data"

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> "CosmosDBIndexStore":
        """Load a CosmosDBIndexStore from a MongoDB URI."""
        mongo_kvstore = MongoDBKVStore.from_uri(uri, db_name)
        return cls(mongo_kvstore, namespace)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> "CosmosDBIndexStore":
        """Load a CosmosDBIndexStore from a MongoDB host and port."""
        mongo_kvstore = MongoDBKVStore.from_host_and_port(host, port, db_name)
        return cls(mongo_kvstore, namespace)
