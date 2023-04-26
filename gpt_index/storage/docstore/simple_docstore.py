import os
from pathlib import Path
from typing import Optional, Union
from gpt_index.storage.docstore.keyval_docstore import KVDocumentStore
from gpt_index.storage.kvstore.simple_kvstore import SimpleKVStore
from gpt_index.storage.kvstore.types import BaseInMemoryKVStore


DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "docstore.json"


class SimpleDocumentStore(KVDocumentStore):
    def __init__(self, simple_kvstore: SimpleKVStore, name_space: Optional[str] = None):
        super().__init__(simple_kvstore, name_space)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: Union[str, Path] = DEFAULT_PERSIST_DIR,
        namespace: Optional[str] = None,
    ):
        persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        simple_kvstore = SimpleKVStore(persist_path)
        return cls(simple_kvstore, namespace)

    def persist(self):
        if isinstance(self._kvstore, BaseInMemoryKVStore):
            self._kvstore.persist()


# alias for backwards compatibility
DocumentStore = SimpleDocumentStore
