from typing import Optional, Any

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE
from llama_index.storage.kvstore.tablestore import TablestoreKVStore


class TablestoreDocumentStore(KVDocumentStore):
    """TablestoreDocument Store.

    Args:
        tablestore_kvstore (TablestoreKVStore): tablestore_kvstore key-value store
        namespace (str): namespace for the docstore

    Returns:
        TablestoreDocumentStore: A Tablestore document store object.
    """

    def __init__(
        self,
        tablestore_kvstore: TablestoreKVStore,
        namespace: str = "llama_index_doc_store_",
        batch_size: int = DEFAULT_BATCH_SIZE,
        node_collection_suffix: str = "data",
        ref_doc_collection_suffix: str = "ref_doc_info",
        metadata_collection_suffix: str = "metadata",
    ) -> None:
        super().__init__(
            kvstore=tablestore_kvstore,
            namespace=namespace,
            batch_size=batch_size,
            node_collection_suffix=node_collection_suffix,
            ref_doc_collection_suffix=ref_doc_collection_suffix,
            metadata_collection_suffix=metadata_collection_suffix,
        )
        self._tablestore_kvstore = tablestore_kvstore

    def clear_all(self):
        doc = self.docs
        self._tablestore_kvstore.delete_all(self._node_collection)
        self._tablestore_kvstore.delete_all(self._ref_doc_collection)
        self._tablestore_kvstore.delete_all(self._metadata_collection)
        for key in doc:
            self.delete_document(doc_id=key)

    @classmethod
    def from_config(
        cls,
        endpoint: Optional[str] = None,
        instance_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        **kwargs: Any,
    ) -> "TablestoreDocumentStore":
        kv_store = TablestoreKVStore(
            endpoint=endpoint,
            instance_name=instance_name,
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            kwargs=kwargs,
        )
        return cls(tablestore_kvstore=kv_store)
