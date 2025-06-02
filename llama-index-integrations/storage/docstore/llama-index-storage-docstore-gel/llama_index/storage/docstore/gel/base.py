from typing import Optional

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE
from llama_index.storage.kvstore.gel import GelKVStore


class GelDocumentStore(KVDocumentStore):
    """
    Gel Document (Node) store.

    A Gel store for Document and Node objects.

    Args:
        gel_kvstore (GelKVStore): Gel key-value store
        namespace (str): namespace for the docstore
        batch_size (int): batch size for bulk operations

    """

    def __init__(
        self,
        gel_kvstore: GelKVStore,
        namespace: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Init a GelDocumentStore."""
        super().__init__(gel_kvstore, namespace=namespace, batch_size=batch_size)
