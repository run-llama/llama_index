import os
from typing import Optional
from llama_index.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.storage.kvstore.simple_kvstore import SimpleKVStore
from llama_index.storage.kvstore.types import BaseInMemoryKVStore
from llama_index.storage.docstore.types import (
    DEFAULT_PERSIST_PATH,
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
)


class SimpleDocumentStore(KVDocumentStore):
    """Simple Document (Node) store.

    An in-memory store for Document and Node objects.

    Args:
        simple_kvstore (SimpleKVStore): simple key-value store
        name_space (str): namespace for the docstore

    """

    def __init__(
        self,
        simple_kvstore: Optional[SimpleKVStore] = None,
        name_space: Optional[str] = None,
    ) -> None:
        """Init a SimpleDocumentStore."""
        simple_kvstore = simple_kvstore or SimpleKVStore()
        super().__init__(simple_kvstore, name_space)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        namespace: Optional[str] = None,
    ) -> "SimpleDocumentStore":
        """Create a SimpleDocumentStore from a persist directory.

        Args:
            persist_dir (str): directory to persist the store
            namespace (Optional[str]): namespace for the docstore

        """

        persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        return cls.from_persist_path(persist_path, namespace=namespace)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        namespace: Optional[str] = None,
    ) -> "SimpleDocumentStore":
        """Create a SimpleDocumentStore from a persist path.

        Args:
            persist_path (str): Path to persist the store
            namespace (Optional[str]): namespace for the docstore

        """

        simple_kvstore = SimpleKVStore.from_persist_path(persist_path)
        return cls(simple_kvstore, namespace)

    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
    ) -> None:
        """Persist the store."""
        if isinstance(self._kvstore, BaseInMemoryKVStore):
            self._kvstore.persist(persist_path)

    @classmethod
    def from_dict(
        cls, save_dict: dict, namespace: Optional[str] = None
    ) -> "SimpleDocumentStore":
        simple_kvstore = SimpleKVStore.from_dict(save_dict)
        return cls(simple_kvstore, namespace)

    def to_dict(self) -> dict:
        assert isinstance(self._kvstore, SimpleKVStore)
        return self._kvstore.to_dict()


# alias for backwards compatibility
DocumentStore = SimpleDocumentStore
