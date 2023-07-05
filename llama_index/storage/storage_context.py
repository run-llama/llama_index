from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fsspec

import llama_index
from llama_index.constants import (
    DOC_STORE_KEY,
    GRAPH_STORE_KEY,
    INDEX_STORE_KEY,
    VECTOR_STORE_KEY,
)
from llama_index.graph_stores.simple import DEFAULT_PERSIST_FNAME as GRAPH_STORE_FNAME
from llama_index.graph_stores.simple import SimpleGraphStore
from llama_index.graph_stores.types import GraphStore
from llama_index.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.storage.docstore.types import DEFAULT_PERSIST_FNAME as DOCSTORE_FNAME
from llama_index.storage.docstore.types import BaseDocumentStore
from llama_index.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.storage.index_store.types import (
    DEFAULT_PERSIST_FNAME as INDEX_STORE_FNAME,
)
from llama_index.storage.index_store.types import BaseIndexStore
from llama_index.vector_stores.simple import DEFAULT_PERSIST_FNAME as VECTOR_STORE_FNAME
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.types import VectorStore
from llama_index.utils import concat_dirs

DEFAULT_PERSIST_DIR = "./storage"


@dataclass
class StorageContext:
    """Storage context.

    The storage context container is a utility container for storing nodes,
    indices, and vectors. It contains the following:
    - docstore: BaseDocumentStore
    - index_store: BaseIndexStore
    - vector_store: VectorStore
    - graph_store: GraphStore

    """

    docstore: BaseDocumentStore
    index_store: BaseIndexStore
    vector_store: VectorStore
    graph_store: GraphStore

    @classmethod
    def from_defaults(
        cls,
        docstore: Optional[BaseDocumentStore] = None,
        index_store: Optional[BaseIndexStore] = None,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        persist_dir: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "StorageContext":
        """Create a StorageContext from defaults.

        Args:
            docstore (Optional[BaseDocumentStore]): document store
            index_store (Optional[BaseIndexStore]): index store
            vector_store (Optional[VectorStore]): vector store
            graph_store (Optional[GraphStore]): graph store

        """
        if persist_dir is None:
            docstore = docstore or SimpleDocumentStore()
            index_store = index_store or SimpleIndexStore()
            vector_store = vector_store or SimpleVectorStore()
            graph_store = graph_store or SimpleGraphStore()
        else:
            docstore = docstore or SimpleDocumentStore.from_persist_dir(
                persist_dir, fs=fs
            )
            index_store = index_store or SimpleIndexStore.from_persist_dir(
                persist_dir, fs=fs
            )
            vector_store = vector_store or SimpleVectorStore.from_persist_dir(
                persist_dir, fs=fs
            )
            graph_store = graph_store or SimpleGraphStore.from_persist_dir(
                persist_dir, fs=fs
            )

        return cls(docstore, index_store, vector_store, graph_store)

    def persist(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        docstore_fname: str = DOCSTORE_FNAME,
        index_store_fname: str = INDEX_STORE_FNAME,
        vector_store_fname: str = VECTOR_STORE_FNAME,
        graph_store_fname: str = GRAPH_STORE_FNAME,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the storage context.

        Args:
            persist_dir (str): directory to persist the storage context

        """
        if fs is not None:
            docstore_path = concat_dirs(persist_dir, docstore_fname)
            index_store_path = concat_dirs(persist_dir, index_store_fname)
            vector_store_path = concat_dirs(persist_dir, vector_store_fname)
            graph_store_path = concat_dirs(persist_dir, graph_store_fname)
        else:
            docstore_path = str(Path(persist_dir) / docstore_fname)
            index_store_path = str(Path(persist_dir) / index_store_fname)
            vector_store_path = str(Path(persist_dir) / vector_store_fname)
            graph_store_path = str(Path(persist_dir) / graph_store_fname)

        self.docstore.persist(persist_path=docstore_path, fs=fs)
        self.index_store.persist(persist_path=index_store_path, fs=fs)
        self.vector_store.persist(persist_path=vector_store_path, fs=fs)
        self.graph_store.persist(persist_path=graph_store_path, fs=fs)

    def to_dict(self) -> dict:
        all_simple = (
            isinstance(self.vector_store, SimpleVectorStore)
            and isinstance(self.docstore, SimpleDocumentStore)
            and isinstance(self.index_store, SimpleIndexStore)
            and isinstance(self.graph_store, SimpleGraphStore)
        )
        if not all_simple:
            raise ValueError(
                "to_dict only available when using simple doc/index/vector stores"
            )

        assert isinstance(self.vector_store, SimpleVectorStore)
        assert isinstance(self.docstore, SimpleDocumentStore)
        assert isinstance(self.index_store, SimpleIndexStore)
        assert isinstance(self.graph_store, SimpleGraphStore)

        return {
            VECTOR_STORE_KEY: self.vector_store.to_dict(),
            DOC_STORE_KEY: self.docstore.to_dict(),
            INDEX_STORE_KEY: self.index_store.to_dict(),
            GRAPH_STORE_KEY: self.graph_store.to_dict(),
        }

    @classmethod
    def from_dict(cls, save_dict: dict) -> "StorageContext":
        """Create a StorageContext from dict."""
        docstore = SimpleDocumentStore.from_dict(save_dict[DOC_STORE_KEY])
        vector_store = SimpleVectorStore.from_dict(save_dict[VECTOR_STORE_KEY])
        index_store = SimpleIndexStore.from_dict(save_dict[INDEX_STORE_KEY])
        graph_store = SimpleGraphStore.from_dict(save_dict[GRAPH_STORE_KEY])
        return cls(
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store,
            graph_store=graph_store,
        )

    def set_global(self) -> "StorageContext":
        """Sets this context as the default storage context for all downstream services except
        when explicitly passed a storage context.
        Changes made to this storage context will affect all downstream services that depend upon it."""
        llama_index.global_service_context = self
        return self

    @classmethod
    def get_global(cls) -> Optional["StorageContext"]:
        """Get the global storage context. Changes made to this global storage context will affect
        all downstream services that depend upon it. The global storage context is by default
        initialized to `StorageContext.from_defaults()`."""
        return llama_index.global_service_context

    @classmethod
    def set_global_to_none(cls):
        """Set the global storage context. When new services are created without an explicit context, it will not
        will not utilize a global context, but instead instantiate a local storage context via `from_defaults`."""
        llama_index.global_storage_context = None


# Set the default storage context as the global storage context
StorageContext.from_defaults().set_global()
