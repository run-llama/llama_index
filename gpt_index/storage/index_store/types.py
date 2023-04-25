from abc import ABC, abstractmethod
from typing import List, Optional

from gpt_index.data_structs.data_structs_v2 import V2IndexStruct


class BaseIndexStore(ABC):
    @abstractmethod
    def index_structs(self) -> List[V2IndexStruct]:
        pass

    @abstractmethod
    def add_index_struct(self, index_struct: V2IndexStruct) -> None:
        pass

    @abstractmethod
    def get_index_struct(self, struct_id: Optional[str] = None) -> V2IndexStruct:
        pass

    def persist(self):
        pass
