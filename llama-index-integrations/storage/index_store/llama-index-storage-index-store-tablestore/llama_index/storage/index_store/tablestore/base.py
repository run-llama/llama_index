from typing import Optional, Any

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore

from llama_index.storage.kvstore.tablestore import TablestoreKVStore


class TablestoreIndexStore(KVIndexStore):
    """
    Tablestore Index store.

    Args:
        tablestore_kvstore (TablestoreKVStore): Tablestore key-value store
        namespace (str): namespace for the index store
        collection_suffix (str): suffix for the table name

    """

    def __init__(
        self,
        tablestore_kvstore: TablestoreKVStore,
        namespace: str = "llama_index_index_store_",
        collection_suffix: str = "data",
    ) -> None:
        """Init a TablestoreIndexStore."""
        super().__init__(
            kvstore=tablestore_kvstore,
            namespace=namespace,
            collection_suffix=collection_suffix,
        )
        self._tablestore_kvstore = tablestore_kvstore

    @classmethod
    def from_config(
        cls,
        endpoint: Optional[str] = None,
        instance_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        **kwargs: Any,
    ) -> "TablestoreIndexStore":
        """Load a TablestoreIndexStore from config."""
        kv_store = TablestoreKVStore(
            endpoint=endpoint,
            instance_name=instance_name,
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            kwargs=kwargs,
        )
        return cls(tablestore_kvstore=kv_store)

    def delete_all_index(self):
        """Delete all index."""
        self._tablestore_kvstore.delete_all(self._collection)
