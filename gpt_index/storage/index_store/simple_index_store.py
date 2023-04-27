import os
from pathlib import Path
from typing import Union
from gpt_index.storage.index_store.keyval_index_store import KVIndexStore
from gpt_index.storage.kvstore.simple_kvstore import SimpleKVStore
from gpt_index.storage.kvstore.types import BaseInMemoryKVStore


DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "index_store.json"


class SimpleIndexStore(KVIndexStore):
    """Simple in-memory Index store.

    Args:
        simple_kvstore (SimpleKVStore): simple key-value store

    """

    def __init__(self, simple_kvstore: SimpleKVStore):
        super().__init__(simple_kvstore)

    @classmethod
    def from_persist_dir(
        cls, persist_dir: str = DEFAULT_PERSIST_DIR
    ) -> "SimpleIndexStore":
        """Create a SimpleIndexStore from a persist directory."""
        persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        simple_kvstore = SimpleKVStore(persist_path)
        return cls(simple_kvstore)

    def persist(self) -> None:
        """Persist the store."""
        if isinstance(self._kvstore, BaseInMemoryKVStore):
            self._kvstore.persist()
