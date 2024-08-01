from typing import List, Optional

from llama_index.legacy.data_structs.data_structs import IndexStruct
from llama_index.legacy.storage.index_store.types import BaseIndexStore
from llama_index.legacy.storage.index_store.utils import (
    index_struct_to_json,
    json_to_index_struct,
)
from llama_index.legacy.storage.kvstore.types import BaseKVStore

DEFAULT_NAMESPACE = "index_store"


class KVIndexStore(BaseIndexStore):
    """Key-Value Index store.

    Args:
        kvstore (BaseKVStore): key-value store
        namespace (str): namespace for the index store

    """

    def __init__(self, kvstore: BaseKVStore, namespace: Optional[str] = None) -> None:
        """Init a KVIndexStore."""
        self._kvstore = kvstore
        self._namespace = namespace or DEFAULT_NAMESPACE
        self._collection = f"{self._namespace}/data"

    def add_index_struct(self, index_struct: IndexStruct) -> None:
        """Add an index struct.

        Args:
            index_struct (IndexStruct): index struct

        """
        key = index_struct.index_id
        data = index_struct_to_json(index_struct)
        self._kvstore.put(key, data, collection=self._collection)

    def delete_index_struct(self, key: str) -> None:
        """Delete an index struct.

        Args:
            key (str): index struct key

        """
        self._kvstore.delete(key, collection=self._collection)

    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        """Get an index struct.

        Args:
            struct_id (Optional[str]): index struct id

        """
        if struct_id is None:
            structs = self.index_structs()
            assert len(structs) == 1
            return structs[0]
        else:
            json = self._kvstore.get(struct_id, collection=self._collection)
            if json is None:
                return None
            return json_to_index_struct(json)

    def index_structs(self) -> List[IndexStruct]:
        """Get all index structs.

        Returns:
            List[IndexStruct]: index structs

        """
        jsons = self._kvstore.get_all(collection=self._collection)
        return [json_to_index_struct(json) for json in jsons.values()]
