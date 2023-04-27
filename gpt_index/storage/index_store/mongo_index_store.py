from typing import Optional
from gpt_index.storage.docstore.keyval_docstore import KVDocumentStore
from gpt_index.storage.index_store.keyval_index_store import KVIndexStore
from gpt_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore


class MongoIndexStore(KVIndexStore):
    def __init__(
        self,
        mongo_kvstore: MongoDBKVStore,
        namespace: Optional[str] = None,
    ):
        super().__init__(mongo_kvstore, namespace=namespace)

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> "MongoIndexStore":
        mongo_kvstore = MongoDBKVStore.from_uri(uri, db_name)
        return cls(mongo_kvstore, namespace)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> "MongoIndexStore":
        mongo_kvstore = MongoDBKVStore.from_host_and_port(host, port, db_name)
        return cls(mongo_kvstore, namespace)
