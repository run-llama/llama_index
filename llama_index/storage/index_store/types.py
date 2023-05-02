from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.data_structs.data_structs_v2 import V2IndexStruct
import os

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "index_store.json"
DEFAULT_PERSIST_PATH = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME)


class BaseIndexStore(ABC):
    @abstractmethod
    def index_structs(self) -> List[V2IndexStruct]:
        pass

    @abstractmethod
    def add_index_struct(self, index_struct: V2IndexStruct) -> None:
        pass

    @abstractmethod
    def delete_index_struct(self, key: str) -> None:
        pass

    @abstractmethod
    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[V2IndexStruct]:
        pass

    def persist(self, persist_path: str = DEFAULT_PERSIST_PATH) -> None:
        """Persist the index store to disk."""
        pass
