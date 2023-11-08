import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import fsspec

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
from llama_index.utils import concat_dirs
from llama_index.vector_stores.simple import DEFAULT_PERSIST_FNAME as VECTOR_STORE_FNAME
from llama_index.vector_stores.simple import (
    DEFAULT_VECTOR_STORE,
    NAMESPACE_SEP,
    SimpleVectorStore,
)
from llama_index.vector_stores.types import VectorStore

DEFAULT_PERSIST_DIR = "./storage"
IMAGE_STORE_FNAME = "image_store.json"


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
    vector_stores: Dict[str, VectorStore]
    graph_store: GraphStore

    @classmethod
    def from_defaults(
        cls,
        docstore: Optional[BaseDocumentStore] = None,
        index_store: Optional[BaseIndexStore] = None,
        vector_store: Optional[VectorStore] = None,
        vector_stores: Optional[Dict[str, VectorStore]] = None,
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
            graph_store = graph_store or SimpleGraphStore()

            if vector_store:
                vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
            else:
                vector_stores = vector_stores or {
                    DEFAULT_VECTOR_STORE: SimpleVectorStore()
                }
        else:
            docstore = docstore or SimpleDocumentStore.from_persist_dir(
                persist_dir, fs=fs
            )
            index_store = index_store or SimpleIndexStore.from_persist_dir(
                persist_dir, fs=fs
            )
            graph_store = graph_store or SimpleGraphStore.from_persist_dir(
                persist_dir, fs=fs
            )

            if vector_store:
                vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
            elif vector_stores:
                vector_stores = vector_stores
            else:
                vector_stores = SimpleVectorStore.from_namespaced_persist_dir(
                    persist_dir, fs=fs
                )

        return cls(
            docstore=docstore,
            index_store=index_store,
            vector_stores=vector_stores,
            graph_store=graph_store,
        )

    def persist(
        self,
        persist_dir: Union[str, os.PathLike] = DEFAULT_PERSIST_DIR,
        docstore_fname: str = DOCSTORE_FNAME,
        index_store_fname: str = INDEX_STORE_FNAME,
        vector_store_fname: str = VECTOR_STORE_FNAME,
        image_store_fname: str = IMAGE_STORE_FNAME,
        graph_store_fname: str = GRAPH_STORE_FNAME,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the storage context.

        Args:
            persist_dir (str): directory to persist the storage context
        """
        if fs is not None:
            persist_dir = str(persist_dir)  # NOTE: doesn't support Windows here
            docstore_path = concat_dirs(persist_dir, docstore_fname)
            index_store_path = concat_dirs(persist_dir, index_store_fname)
            graph_store_path = concat_dirs(persist_dir, graph_store_fname)
        else:
            persist_dir = Path(persist_dir)
            docstore_path = str(persist_dir / docstore_fname)
            index_store_path = str(persist_dir / index_store_fname)
            graph_store_path = str(persist_dir / graph_store_fname)

        self.docstore.persist(persist_path=docstore_path, fs=fs)
        self.index_store.persist(persist_path=index_store_path, fs=fs)
        self.graph_store.persist(persist_path=graph_store_path, fs=fs)

        # save each vector store under it's namespace
        for vector_store_name, vector_store in self.vector_stores.items():
            if fs is not None:
                vector_store_path = concat_dirs(
                    str(persist_dir),
                    f"{vector_store_name}{NAMESPACE_SEP}{vector_store_fname}",
                )
            else:
                vector_store_path = str(
                    Path(persist_dir)
                    / f"{vector_store_name}{NAMESPACE_SEP}{vector_store_fname}"
                )

            vector_store.persist(persist_path=vector_store_path, fs=fs)

    def to_dict(self) -> dict:
        all_simple = (
            isinstance(self.docstore, SimpleDocumentStore)
            and isinstance(self.index_store, SimpleIndexStore)
            and isinstance(self.graph_store, SimpleGraphStore)
            and all(
                isinstance(vs, SimpleVectorStore) for vs in self.vector_stores.values()
            )
        )
        if not all_simple:
            raise ValueError(
                "to_dict only available when using simple doc/index/vector stores"
            )

        assert isinstance(self.docstore, SimpleDocumentStore)
        assert isinstance(self.index_store, SimpleIndexStore)
        assert isinstance(self.graph_store, SimpleGraphStore)

        return {
            VECTOR_STORE_KEY: {
                key: vector_store.to_dict()
                for key, vector_store in self.vector_stores.items()
                if isinstance(vector_store, SimpleVectorStore)
            },
            DOC_STORE_KEY: self.docstore.to_dict(),
            INDEX_STORE_KEY: self.index_store.to_dict(),
            GRAPH_STORE_KEY: self.graph_store.to_dict(),
        }

    @classmethod
    def from_dict(cls, save_dict: dict) -> "StorageContext":
        """Create a StorageContext from dict."""
        docstore = SimpleDocumentStore.from_dict(save_dict[DOC_STORE_KEY])
        index_store = SimpleIndexStore.from_dict(save_dict[INDEX_STORE_KEY])
        graph_store = SimpleGraphStore.from_dict(save_dict[GRAPH_STORE_KEY])

        vector_stores: Dict[str, VectorStore] = {}
        for key, vector_store_dict in save_dict[VECTOR_STORE_KEY].items():
            vector_stores[key] = SimpleVectorStore.from_dict(vector_store_dict)

        return cls(
            docstore=docstore,
            index_store=index_store,
            vector_stores=vector_stores,
            graph_store=graph_store,
        )

    @property
    def vector_store(self) -> VectorStore:
        """Backwrds compatibility for vector_store property."""
        return self.vector_stores[DEFAULT_VECTOR_STORE]

    def add_vector_store(self, vector_store: VectorStore, namespace: str) -> None:
        """Add a vector store to the storage context."""
        self.vector_stores[namespace] = vector_store
