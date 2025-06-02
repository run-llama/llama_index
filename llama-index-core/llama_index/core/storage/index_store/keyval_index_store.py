from typing import List, Optional

from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.storage.index_store.utils import (
    index_struct_to_json,
    json_to_index_struct,
)
from llama_index.core.storage.kvstore.types import BaseKVStore

DEFAULT_NAMESPACE = "index_store"
DEFAULT_COLLECTION_SUFFIX = "/data"


class KVIndexStore(BaseIndexStore):
    """
    Key-Value Index store.

    Args:
        kvstore (BaseKVStore): key-value store
        namespace (str): namespace for the index store
        collection_suffix (str): suffix for the collection name

    """

    def __init__(
        self,
        kvstore: BaseKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a KVIndexStore."""
        self._kvstore = kvstore
        self._namespace = namespace or DEFAULT_NAMESPACE
        self._collection_suffix = collection_suffix or DEFAULT_COLLECTION_SUFFIX
        self._collection = f"{self._namespace}{self._collection_suffix}"

    def add_index_struct(self, index_struct: IndexStruct) -> None:
        """
        Add an index struct.

        Args:
            index_struct (IndexStruct): index struct

        """
        key = index_struct.index_id
        data = index_struct_to_json(index_struct)
        self._kvstore.put(key, data, collection=self._collection)

    def delete_index_struct(self, key: str) -> None:
        """
        Delete an index struct.

        Args:
            key (str): index struct key

        """
        self._kvstore.delete(key, collection=self._collection)

    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        """
        Get an index struct.

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
        """
        Get all index structs.

        Returns:
            List[IndexStruct]: index structs

        """
        jsons = self._kvstore.get_all(collection=self._collection)
        return [json_to_index_struct(json) for json in jsons.values()]

    async def async_add_index_struct(self, index_struct: IndexStruct) -> None:
        """
        Asynchronously add an index struct.

        Args:
            index_struct (IndexStruct): index struct

        """
        key = index_struct.index_id
        data = index_struct_to_json(index_struct)
        await self._kvstore.aput(key, data, collection=self._collection)

    async def adelete_index_struct(self, key: str) -> None:
        """
        Asynchronously delete an index struct.

        Args:
            key (str): index struct key

        """
        await self._kvstore.adelete(key, collection=self._collection)

    async def aget_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        """
        Asynchronously get an index struct.

        Args:
            struct_id (Optional[str]): index struct id

        """
        if struct_id is None:
            structs = await self.async_index_structs()
            assert len(structs) == 1
            return structs[0]
        else:
            json = await self._kvstore.aget(struct_id, collection=self._collection)
            if json is None:
                return None
            return json_to_index_struct(json)

    async def async_index_structs(self) -> List[IndexStruct]:
        """
        Asynchronously get all index structs.

        Returns:
            List[IndexStruct]: index structs

        """
        jsons = await self._kvstore.aget_all(collection=self._collection)
        return [json_to_index_struct(json) for json in jsons.values()]
