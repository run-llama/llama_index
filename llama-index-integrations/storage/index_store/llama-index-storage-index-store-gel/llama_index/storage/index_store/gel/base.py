from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.gel import GelKVStore


class GelIndexStore(KVIndexStore):
    """
    Gel Index store.

    Args:
        gel_kvstore (GelKVStore): Gel key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        gel_kvstore: GelKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a GelIndexStore."""
        super().__init__(
            gel_kvstore, namespace=namespace, collection_suffix=collection_suffix
        )
