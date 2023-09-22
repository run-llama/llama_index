from typing import List, Optional
from llama_index.data_structs.data_structs import IndexStruct
from llama_index.storage.index_store.types import BaseIndexStore
from llama_index.constants import DATA_KEY, TYPE_KEY
from llama_index.storage.kvstore.simple_pickled_kvstore import SimplePickledKVStore

DEFAULT_NAMESPACE = "index_store"


class PickledKVIndexStore(SimplePickledKVStore):
    """Key-Value Index store that uses pickle as a storing mechanism.

    Args:
        kvstore (BaseKVStore): key-value store
        namespace (str): namespace for the index store

    """

    def __init__(self, kvstore: SimplePickledKVStore, namespace: Optional[str] = None) -> None:
        """Init a PickledKVIndexStore."""
        self._kvstore = kvstore
        self._namespace = namespace or DEFAULT_NAMESPACE
        self._collection = f"{self._namespace}/data"

    def add_index_struct(self, index_struct: IndexStruct) -> None:
        """Add an index struct.

        Args:
            index_struct (IndexStruct): index struct

        """
        key = index_struct.index_id
        data = {
            TYPE_KEY: index_struct.get_type(),
            DATA_KEY: index_struct,
        }
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
            data = self._kvstore.get(struct_id, collection=self._collection)
            if data is None:
                return None
            return data[DATA_KEY]

    def index_structs(self) -> List[IndexStruct]:
        """Get all index structs.

        Returns:
            List[IndexStruct]: index structs

        """
        all_data = self._kvstore.get_all(collection=self._collection)
        return [data[DATA_KEY] for data in all_data.values()]
