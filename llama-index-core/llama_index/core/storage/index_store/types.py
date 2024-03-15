import os
from abc import ABC, abstractmethod
from typing import List, Optional

import fsspec
from llama_index.core.data_structs.data_structs import IndexStruct

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "index_store.json"
DEFAULT_PERSIST_PATH = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME)


class BaseIndexStore(ABC):
    @abstractmethod
    def index_structs(self) -> List[IndexStruct]:
        pass

    @abstractmethod
    def add_index_struct(self, index_struct: IndexStruct) -> None:
        pass

    @abstractmethod
    def delete_index_struct(self, key: str) -> None:
        pass

    @abstractmethod
    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        pass

    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the index store to disk."""
