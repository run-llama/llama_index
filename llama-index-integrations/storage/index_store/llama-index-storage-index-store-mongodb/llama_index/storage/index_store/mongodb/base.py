from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.mongodb import MongoDBKVStore


class MongoIndexStore(KVIndexStore):
    """
    Mongo Index store.

    Args:
        mongo_kvstore (MongoDBKVStore): MongoDB key-value store
        namespace (str): namespace for the index store
        collection_suffix (str): suffix for the collection name

    """

    def __init__(
        self,
        mongo_kvstore: MongoDBKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a MongoIndexStore."""
        super().__init__(
            mongo_kvstore, namespace=namespace, collection_suffix=collection_suffix
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "MongoIndexStore":
        """Load a MongoIndexStore from a MongoDB URI."""
        mongo_kvstore = MongoDBKVStore.from_uri(uri, db_name)
        return cls(mongo_kvstore, namespace, collection_suffix)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "MongoIndexStore":
        """Load a MongoIndexStore from a MongoDB host and port."""
        mongo_kvstore = MongoDBKVStore.from_host_and_port(host, port, db_name)
        return cls(mongo_kvstore, namespace, collection_suffix)
