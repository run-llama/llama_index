from dataclasses import dataclass
import pathlib
from typing import Optional, Union
from gpt_index.storage.docstore.simple_docstore import SimpleDocumentStore
from gpt_index.storage.docstore.types import BaseDocumentStore
from gpt_index.storage.index_store.simple_index_store import SimpleIndexStore
from gpt_index.storage.index_store.types import BaseIndexStore
from gpt_index.vector_stores.simple import SimpleVectorStore
from gpt_index.vector_stores.types import VectorStore

DEFAULT_PERSIST_DIR = "./storage"


@dataclass
class StorageContext:
    """Storage context.

    The storage context container is a utility container for storing nodes,
    indices, and vectors. It contains the following:
    - docstore: BaseDocumentStore
    - index_store: BaseIndexStore
    - vector_store: VectorStore

    """

    docstore: BaseDocumentStore
    index_store: BaseIndexStore
    vector_store: VectorStore

    @classmethod
    def from_defaults(
        cls,
        docstore: Optional[BaseDocumentStore] = None,
        index_store: Optional[BaseIndexStore] = None,
        vector_store: Optional[VectorStore] = None,
        persist_dir: str = DEFAULT_PERSIST_DIR,
    ) -> "StorageContext":
        """Create a StorageContext from defaults.

        Args:
            docstore (Optional[BaseDocumentStore]): document store
            index_store (Optional[BaseIndexStore]): index store
            vector_store (Optional[VectorStore]): vector store

        """
        docstore = docstore or SimpleDocumentStore.from_persist_dir(persist_dir)
        index_store = index_store or SimpleIndexStore.from_persist_dir(persist_dir)
        vector_store = vector_store or SimpleVectorStore.from_persist_dir(persist_dir)
        return cls(docstore, index_store, vector_store)

    def persist(self) -> None:
        self.docstore.persist()
        self.index_store.persist()
        self.vector_store.persist()
