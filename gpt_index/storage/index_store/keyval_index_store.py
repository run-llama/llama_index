from typing import List, Optional
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.storage.index_store.types import BaseIndexStore
from gpt_index.storage.index_store.utils import (
    index_struct_to_json,
    json_to_index_struct,
)
from gpt_index.storage.kvstore.types import BaseKVStore


class KeyValIndexStore(BaseIndexStore):
    def __init__(self, kvstore: BaseKVStore) -> None:
        self._kvstore = kvstore

    def add_index_struct(self, index_struct: V2IndexStruct) -> None:
        key = index_struct.index_id
        data = index_struct_to_json(index_struct)
        self._kvstore.add(key, data)

    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[V2IndexStruct]:
        if struct_id is None:
            structs = self.index_structs()
            assert len(structs) == 1
            return structs[0]
        else:
            json = self._kvstore.get(struct_id)
            if json is None:
                return None
            return json_to_index_struct(json)

    def index_structs(self) -> List[V2IndexStruct]:
        jsons = self._kvstore.get_all()
        return [json_to_index_struct(json) for json in jsons]
