from typing import Optional

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE
from llama_index.storage.kvstore.duckdb import DuckDBKVStore


class DuckDBDocumentStore(KVDocumentStore):
    """
    DuckDB Document (Node) store.

    A DuckDB store for Document and Node objects.

    Args:
        duckdb_kvstore (DuckDBKVStore): DuckDB key-value store
        namespace (str): namespace for the docstore

    """

    def __init__(
        self,
        duckdb_kvstore: DuckDBKVStore,
        namespace: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Init a DuckDBDocumentStore."""
        super().__init__(duckdb_kvstore, namespace=namespace, batch_size=batch_size)
        # avoid conflicts with duckdb index store
        self._node_collection = f"{self._namespace}/doc"
