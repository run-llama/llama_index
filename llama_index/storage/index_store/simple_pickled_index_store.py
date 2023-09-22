import os
from typing import Optional
import fsspec
from llama_index.storage.index_store.pickled_keyval_index_store import PickledKVIndexStore
from llama_index.storage.kvstore.simple_pickled_kvstore import SimplePickledKVStore
from llama_index.storage.index_store.types import DEFAULT_PERSIST_DIR, DEFAULT_PICKLE_FNAME, DEFAULT_PICKLE_PERSIST_PATH
from llama_index.utils import concat_dirs


class SimplePickledIndexStore(PickledKVIndexStore):
    """Simple in-memory Index store that uses pickles as a storing mechanism.

    Args:
        simple_pickled_kvstore (PickledKVIndexStore): simple pickled key-value store

    """

    def __init__(
        self,
        simple_pickled_kvstore: Optional[PickledKVIndexStore] = None,
    ) -> None:
        """Init a SimplePickledIndexStore."""
        simple_pickled_kvstore = simple_pickled_kvstore or PickledKVIndexStore()
        super().__init__(simple_pickled_kvstore)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimplePickledIndexStore":
        """Create a SimplePickledIndexStore from a persist directory."""
        if fs is not None:
            persist_path = concat_dirs(persist_dir, DEFAULT_PICKLE_FNAME)
        else:
            persist_path = os.path.join(persist_dir, DEFAULT_PICKLE_FNAME)
        return cls.from_persist_path(persist_path, fs=fs)
    
    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimplePickledIndexStore":
        """Create a SimpleIndexStore from a persist path."""
        fs = fs or fsspec.filesystem("file")
        simple_kvstore = SimplePickledKVStore.from_persist_path(persist_path, fs=fs)
        return cls(simple_kvstore)
    
    def persist(
        self,
        persist_path: str = DEFAULT_PICKLE_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the store."""
        if isinstance(self._kvstore, PickledKVIndexStore):
            self._kvstore.persist(persist_path, fs=fs)
    
    @classmethod
    def from_dict(cls, save_dict: dict) -> "SimplePickledIndexStore":
        simple_kvstore = SimplePickledKVStore.from_dict(save_dict)
        return cls(simple_kvstore)

    def to_dict(self) -> dict:
        assert isinstance(self._kvstore, SimplePickledKVStore)
        return self._kvstore.to_dict()
