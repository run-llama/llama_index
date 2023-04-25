from typing import Optional
from gpt_index.storage.docstore.keyval_docstore import KeyValDocumentStore
from gpt_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore


class MongoDocumentStore(KeyValDocumentStore):
    def __init__(
        self,
        mongo_keyval_store: MongoDBKVStore,
        namespace: Optional[str] = None,
    ):
        super().__init__(mongo_keyval_store, namespace)

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> "MongoDocumentStore":
        mongo_keyval_store = MongoDBKVStore.from_uri(uri, db_name)
        return cls(mongo_keyval_store, namespace)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> "MongoDocumentStore":
        mongo_keyval_store = MongoDBKVStore.from_host_and_port(host, port, db_name)
        return cls(mongo_keyval_store, namespace)
