import os
from typing import Optional
from gpt_index.storage.docstore.keyval_docstore import KeyValDocumentStore
from gpt_index.storage.kvstore.simple_kvstore import SimpleKVStore
from gpt_index.storage.kvstore.types import BaseInMemoryKVStore


DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "docstore.json"


class SimpleDocumentStore(KeyValDocumentStore):
    def __init__(
        self, simple_keyval_store: SimpleKVStore, name_space: Optional[str] = None
    ):
        super().__init__(simple_keyval_store, name_space)

    @classmethod
    def from_persist_dir(
        cls, persist_dir: str = DEFAULT_PERSIST_DIR, namespace: Optional[str] = None
    ):
        persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        simple_keyval_store = SimpleKVStore(persist_path, namespace)
        return cls(simple_keyval_store)

    def persist(self):
        if isinstance(self._keyval_store, BaseInMemoryKVStore):
            self._keyval_store.persist()


# alias for backwards compatibility
DocumentStore = SimpleDocumentStore
