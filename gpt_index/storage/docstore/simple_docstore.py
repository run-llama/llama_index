import os
from gpt_index.storage.docstore.keyval_docstore import KeyValDocumentStore
from gpt_index.storage.keyval_store.simple import SimpleKeyValStore
from gpt_index.storage.keyval_store.types import BaseInMemoryKeyValStore


DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "docstore.json"


class SimpleDocumentStore(KeyValDocumentStore):
    def __init__(self, simple_keyval_store: SimpleKeyValStore):
        super().__init__(simple_keyval_store)

    @classmethod
    def from_persist_dir(cls, persist_dir: str = DEFAULT_PERSIST_DIR):
        persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        simple_keyval_store = SimpleKeyValStore(persist_path)
        return cls(simple_keyval_store)

    def persist(self):
        assert isinstance(self._keyval_store, BaseInMemoryKeyValStore)
        self._keyval_store.persist()


# alias for backwards compatibility
DocumentStore = SimpleDocumentStore
