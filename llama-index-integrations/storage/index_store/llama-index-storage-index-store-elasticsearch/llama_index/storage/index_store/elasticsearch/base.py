from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.elasticsearch import ElasticsearchKVStore


class ElasticsearchIndexStore(KVIndexStore):
    """Elasticsearch Index store.

    Args:
        elasticsearch_kvstore (ElasticsearchKVStore): Elasticsearch key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        elasticsearch_kvstore: ElasticsearchKVStore,
        namespace: Optional[str] = None,
    ) -> None:
        """Init a ElasticsearchIndexStore."""
        super().__init__(elasticsearch_kvstore, namespace=namespace)
