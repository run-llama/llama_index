from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import (
    KVIndexStore,
    DEFAULT_COLLECTION_SUFFIX,
)
from llama_index.storage.kvstore.duckdb import DuckDBKVStore


class DuckDBIndexStore(KVIndexStore):
    """
    DuckDB Index store.

    Args:
        duckdb_kvstore (DuckDBKVStore): DuckDB key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        duckdb_kvstore: DuckDBKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a DuckDBIndexStore."""
        super().__init__(
            duckdb_kvstore, namespace=namespace, collection_suffix=collection_suffix
        )
        # avoid conflicts with duckdb docstore
        if self._collection.endswith(DEFAULT_COLLECTION_SUFFIX):
            self._collection = f"{self._namespace}/index"
