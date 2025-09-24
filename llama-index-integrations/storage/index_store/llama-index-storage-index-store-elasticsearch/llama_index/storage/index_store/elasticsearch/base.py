from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.elasticsearch import ElasticsearchKVStore


class ElasticsearchIndexStore(KVIndexStore):
    """
    Elasticsearch Index store.

    Args:
        elasticsearch_kvstore (ElasticsearchKVStore): Elasticsearch key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        elasticsearch_kvstore: ElasticsearchKVStore,
        collection_index: Optional[str] = None,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a ElasticsearchIndexStore."""
        super().__init__(
            elasticsearch_kvstore,
            namespace=namespace,
            collection_suffix=collection_suffix,
        )
        if collection_index:
            self._collection = collection_index
        else:
            self._collection = f"llama_index-index_store.data-{self._namespace}"
