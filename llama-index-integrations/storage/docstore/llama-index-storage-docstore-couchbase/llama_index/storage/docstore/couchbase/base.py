from typing import Optional
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.storage.kvstore.couchbase import CouchbaseKVStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE


class CouchbaseDocumentStore(KVDocumentStore):
    """Couchbase Document (Node) store.
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
        """Init a CouchbaseDocumentStore."""
        super().__init__(
            couchbase_kvstore,
            namespace=namespace,
            batch_size=batch_size,
            node_collection_suffix=node_collection_suffix,
            ref_doc_collection_suffix=ref_doc_collection_suffix,
            metadata_collection_suffix=metadata_collection_suffix,
        )
