from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.couchbase import CouchbaseKVStore


class CouchbaseIndexStore(KVIndexStore):
    """Couchbase Index store."""

    def __init__(
        self,
        couchbase_kvstore: CouchbaseKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Initialize a CouchbaseIndexStore.

        Args:
        couchbase_kvstore (CouchbaseKVStore): Couchbase key-value store
        namespace (str): namespace for the index store
        collection_suffix (str): suffix for the collection name
        """
        super().__init__(
            couchbase_kvstore,
            namespace=namespace,
            collection_suffix=collection_suffix,
        )
