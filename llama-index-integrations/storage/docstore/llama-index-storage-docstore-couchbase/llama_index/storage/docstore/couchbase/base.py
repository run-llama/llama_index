from typing import Any, Optional
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.storage.kvstore.couchbase import CouchbaseKVStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE


class CouchbaseDocumentStore(KVDocumentStore):
    """
    Couchbase Document (Node) store.
    A documents store for Document and Node objects using Couchbase.
    """

    def __init__(
        self,
        couchbase_kvstore: CouchbaseKVStore,
        namespace: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
    ) -> None:
        """
        Initialize a CouchbaseDocumentStore.

        Args:
            couchbase_kvstore (CouchbaseKVStore): Couchbase key-value store
            namespace (str): namespace for the docstore
            batch_size (int): batch size for fetching documents
            node_collection_suffix (str): suffix for the node collection
            ref_doc_collection_suffix (str): suffix for the  Refdoc collection
            metadata_collection_suffix (str): suffix for the metadata collection

        """
        super().__init__(
            couchbase_kvstore,
            namespace=namespace,
            batch_size=batch_size,
            node_collection_suffix=node_collection_suffix,
            ref_doc_collection_suffix=ref_doc_collection_suffix,
            metadata_collection_suffix=metadata_collection_suffix,
        )

    @classmethod
    def from_couchbase_client(
        cls,
        client: Any,
        bucket_name: str,
        scope_name: str,
        namespace: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        async_client: Optional[Any] = None,
    ) -> "CouchbaseDocumentStore":
        """Initialize a CouchbaseDocumentStore from a Couchbase client."""
        couchbase_kvstore = CouchbaseKVStore.from_couchbase_client(
            client=client,
            bucket_name=bucket_name,
            scope_name=scope_name,
            async_client=async_client,
        )
        return cls(
            couchbase_kvstore,
            namespace,
            batch_size,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
        )
