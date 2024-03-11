from typing import Optional

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE
from llama_index.storage.kvstore.elasticsearch import ElasticsearchKVStore 


class ElasticsearchDocumentStore(KVDocumentStore):
    """Elasticsearch Document (Node) store.

    An Elasticsearch store for Document and Node objects.

    Args:
        es_kvstore (ElasticsearchKVStore): Elasticsearch key-value store
        namespace (str): namespace for the docstore

    """

    def __init__(
        self,
        es_kvstore: ElasticsearchKVStore,
        namespace: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Init a ElasticDocumentStore."""
        super().__init__(es_kvstore, namespace=namespace, batch_size=batch_size)

    # @classmethod
    # def from_uri(
    #     cls,
    #     uri: str,
    #     db_name: Optional[str] = None,
    #     namespace: Optional[str] = None,
    # ) -> "MongoDocumentStore":
    #     """Load a MongoDocumentStore from a MongoDB URI."""
    #     mongo_kvstore = MongoDBKVStore.from_uri(uri, db_name)
    #     return cls(mongo_kvstore, namespace)

    # @classmethod
    # def from_host_and_port(
    #     cls,
    #     host: str,
    #     port: int,
    #     db_name: Optional[str] = None,
    #     namespace: Optional[str] = None,
    # ) -> "MongoDocumentStore":
    #     """Load a MongoDocumentStore from a MongoDB host and port."""
    #     mongo_kvstore = MongoDBKVStore.from_host_and_port(host, port, db_name)
    #     return cls(mongo_kvstore, namespace)
