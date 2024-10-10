from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.couchbase import CouchbaseKVStore


class CouchbaseIndexStore(KVIndexStore):
    """Couchbase Index store.

    Args:
        mongo_kvstore (MongoDBKVStore): MongoDB key-value store
        namespace (str): namespace for the index store
        collection_suffix (str): suffix for the collection name

    """

    def __init__(
        self,
        couchbase_kvstore: CouchbaseKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a CouchbaseDocumentStore."""
        super().__init__(
            couchbase_kvstore,
            namespace=namespace,
            collection_suffix=collection_suffix,
        )
