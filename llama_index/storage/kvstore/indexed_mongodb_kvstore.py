from typing import Any, Dict, List, Optional, Tuple

from llama_index.storage.kvstore.mongodb_kvstore import IMPORT_ERROR_MSG, MongoDBKVStore
from llama_index.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
)


class IndexedMongoDBKVStore(MongoDBKVStore):
    """Indexed MongoDB Key-Value store.

    Allows for querying of objects stored in the Mongo KV store by
    attributes other than the given key. Important for Mongo in particular,
    as you
    Requires the collection_name to be defined upfront.

    Args:
        mongo_client (Any): MongoDB client
        uri (Optional[str]): MongoDB URI
        host (Optional[str]): MongoDB host
        port (Optional[int]): MongoDB port
        db_name (Optional[str]): MongoDB database name
        collection_name (Optional[str]): MongoDB collection name
        indexed_attributes (Optional[List[str]]): List of attributes to index

    """

    def __init__(
        self,
        mongo_client: Any,
        mongo_aclient: Optional[Any] = None,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        indexed_attributes: Optional[List[str]] = None,
    ) -> None:
        """Init a IndexedMongoDBKVStore."""
        super().__init__(
            mongo_client=mongo_client,
            mongo_aclient=mongo_aclient,
            uri=uri,
            host=host,
            port=port,
            db_name=db_name,
        )

        if indexed_attributes is None:
            indexed_attributes = []

        self._collection_name = collection_name or DEFAULT_COLLECTION
        self._indexed_attributes = indexed_attributes or []
        self._created_indexes = False

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        indexed_attributes: Optional[List[str]] = None,
    ) -> "IndexedMongoDBKVStore":
        """Load a IndexedMongoDBKVStore from a MongoDB URI.

        Args:
            uri (str): MongoDB URI
            db_name (Optional[str]): MongoDB database name
            collection_name (Optional[str]): MongoDB collection name
            indexed_attributes (Optional[List[str]]): List of attributes to index

        """
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        mongo_client: MongoClient = MongoClient(uri)
        mongo_aclient: AsyncIOMotorClient = AsyncIOMotorClient(uri)
        return cls(
            mongo_client=mongo_client,
            mongo_aclient=mongo_aclient,
            db_name=db_name,
            uri=uri,
            collection_name=collection_name,
            indexed_attributes=indexed_attributes,
        )

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        indexed_attributes: Optional[List[str]] = None,
    ) -> "IndexedMongoDBKVStore":
        """Load a IndexedMongoDBKVStore from a MongoDB host and port.

        Args:
            host (str): MongoDB host
            port (int): MongoDB port
            db_name (Optional[str]): MongoDB database name
            collection_name (Optional[str]): MongoDB collection name
            indexed_attributes (Optional[List[str]]): List of attributes to index

        """
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        mongo_client: MongoClient = MongoClient(host, port)
        mongo_aclient: AsyncIOMotorClient = AsyncIOMotorClient(host, port)
        return cls(
            mongo_client=mongo_client,
            mongo_aclient=mongo_aclient,
            db_name=db_name,
            host=host,
            port=port,
            collection_name=collection_name,
            indexed_attributes=indexed_attributes,
        )

    def create_indexes(self) -> None:
        """Create index on the indexed attributes in the predefined collection_name."""
        if not self._created_indexes:
            for attr in self._indexed_attributes:
                self._db[self._collection_name].create_index(attr)
            self._created_indexes = True

    async def acreate_indexes(self) -> None:
        """Create index on the indexed attributes in the predefined collection_name.

        Same as create_indexes but uses the async mongo_aclient from motor.
        """
        if not self._created_indexes:
            self._check_async_client()
            for attr in self._indexed_attributes:
                await self._adb[self._collection_name].create_index(attr)
            self._created_indexes = True

    def _assert_collection_name_matches(self, collection: Optional[str]) -> str:
        """
        Assert the collection name matches the expected collection name set in the __init__ constructor.
        """
        if collection is not None:
            assert self._collection_name == collection, (
                f"Collection name {collection} does not match the "
                f"predefined collection name {self._collection_name}"
            )
        return self._collection_name

    def put(
        self,
        key: str,
        val: dict,
        collection: Optional[str] = None,
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (Optional[str]): collection name

        """
        self.create_indexes()
        collection = self._assert_collection_name_matches(collection)
        return super().put(key, val, collection=collection)

    async def aput(
        self,
        key: str,
        val: dict,
        collection: Optional[str] = None,
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (Optional[str]): collection name

        """
        await self.acreate_indexes()
        collection = self._assert_collection_name_matches(collection)
        return await super().aput(key, val, collection=collection)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.create_indexes()
        collection = self._assert_collection_name_matches(collection)

        return super().put_all(kv_pairs, collection=collection, batch_size=batch_size)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        await self.acreate_indexes()
        collection = self._assert_collection_name_matches(collection)

        return await super().aput_all(
            kv_pairs, collection=collection, batch_size=batch_size
        )

    def get(self, key: str, collection: Optional[str] = None) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        self.create_indexes()
        collection = self._assert_collection_name_matches(collection)
        return super().get(key, collection=collection)

    async def aget(self, key: str, collection: Optional[str] = None) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        await self.acreate_indexes()
        collection = self._assert_collection_name_matches(collection)
        return await super().aget(key, collection=collection)

    def get_all(self, collection: Optional[str] = None) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (Optional[str]): collection name

        """
        self.create_indexes()
        collection = self._assert_collection_name_matches(collection)
        return super().get_all(collection=collection)

    def get_all_with_filters(
        self, filters: Dict[str, Any], collection: Optional[str] = None
    ) -> Dict[str, dict]:
        """Get all values from the store that match the filters.

        Args:
            filters (dict): filters
            collection (Optional[str]): collection name

        """
        self.create_indexes()
        collection = self._assert_collection_name_matches(collection)
        results = self._db[collection].find(filters)
        output = {}
        for result in results:
            key = result.pop("_id")
            output[key] = result
        return output

    async def aget_all(self, collection: Optional[str] = None) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (Optional[str]): collection name

        """
        await self.acreate_indexes()
        collection = self._assert_collection_name_matches(collection)

        return await super().aget_all(collection=collection)

    async def aget_all_with_filters(
        self,
        filters: Dict[str, Any],
        collection: Optional[str] = None,
    ) -> Dict[str, dict]:
        """Get all values from the store that match the filters.

        Args:
            filters (dict): filters
            collection (Optional[str]): collection name

        """
        await self.acreate_indexes()
        collection = self._assert_collection_name_matches(collection)

        results = self._adb[collection].find(filters)
        output = {}
        for result in await results.to_list(length=None):
            key = result.pop("_id")
            output[key] = result
        return output

    def delete(self, key: str, collection: Optional[str] = None) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (Optional[str]): collection name

        """
        self.create_indexes()
        collection = self._assert_collection_name_matches(collection)
        return super().delete(key, collection=collection)

    async def adelete(self, key: str, collection: Optional[str] = None) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (Optional[str]): collection name

        """
        await self.acreate_indexes()
        collection = self._assert_collection_name_matches(collection)

        return await super().adelete(key, collection=collection)
