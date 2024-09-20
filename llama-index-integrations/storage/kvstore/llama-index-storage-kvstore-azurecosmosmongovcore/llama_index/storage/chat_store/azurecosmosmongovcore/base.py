import logging
from abc import ABC
from typing import Dict, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.storage.kvstore.types import BaseKVStore, DEFAULT_COLLECTION
from pymongo import MongoClient

logger = logging.getLogger(__name__)

APP_NAME = "Llama-Index-CDBMongoVCore-KVStore-Python"


class AzureCosmosMongoVCoreKVStore(BaseKVStore, ABC):
    """Creates an AzureCosmosMongoVCoreKVStore."""

    _mongo_client = MongoClient = PrivateAttr()
    _database = DatabaseProxy = PrivateAttr()
    _collection = ContainerProxy = PrivateAttr()

    def __init__(
        self,
        mongo_client: MongoClient,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self._mongo_client = mongo_client
        self._uri = uri
        self._host = host
        self._port = port
        self._database = self._mongo_client[db_name]
        self._collection = self._mongo_client[db_name][collection_name]

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "AzureCosmosMongoVCoreKVStore":
        """Creates an instance of AzureCosmosMongoVCoreKVStore using a connection string."""
        mongo_client = MongoClient(connection_string, appname=APP_NAME)

        return cls(
            mongo_client=mongo_client,
            db_name=db_name,
            collection_name=collection_name,
        )

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "AzureCosmosMongoVCoreKVStore":
        """Initializes AzureCosmosMongoVCoreKVStore from an endpoint url and key."""
        mongo_client = MongoClient(host=host, port=port, appname=APP_NAME)

        return cls(
            mongo_client=mongo_client,
            host=host,
            port=port,
            db_name=db_name,
            collection_name=collection_name,
        )

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name
        """
        self._collection.updateOne(
            {"_id": key}, {"$set": {"messages": val}}, upsert=True
        )

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name
        """
        raise NotImplementedError

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        response = self._collection.find_one({"_id": key})
        if response is not None:
            messages = response.get("messages")
        else:
            messages = {}
        return messages

    async def aget(self, key: str, collection: str = DEFAULT_COLLECTION) -> dict | None:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name
        """
        raise NotImplementedError

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (str): collection name

        """
        items = self._collection.find()
        output = {}
        for item in items:
            key = item.pop("id")
            output[key] = item
        return output

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (str): collection name
        """
        raise NotImplementedError

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        result = self._collection.delete_one({"_id": key})
        return result.deleted_count > 0

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name
        """
        raise NotImplementedError

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AzureCosmosMongoVCoreKVStore"
