from typing import Optional
from gpt_index.storage.docstore.keyval_docstore import KeyValDocumentStore
from gpt_index.storage.keyval_store.mongodb import MongoDBKeyValStore


class MongoDocumentStore(KeyValDocumentStore):
    def __init__(self, mongo_keyval_store: MongoDBKeyValStore):
        super().__init__(mongo_keyval_store)

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "MongoDocumentStore":
        mongo_keyval_store = MongoDBKeyValStore.from_uri(uri, db_name, collection_name)
        return cls(mongo_keyval_store)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "MongoDocumentStore":
        mongo_keyval_store = MongoDBKeyValStore.from_host_and_port(
            host, port, db_name, collection_name
        )
        return cls(mongo_keyval_store)
