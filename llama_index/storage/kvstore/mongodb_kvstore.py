from typing import Any, Dict, Optional, cast

from llama_index.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore

IMPORT_ERROR_MSG = "`pymongo` package not found, please run `pip install pymongo`"


class MongoDBKVStore(BaseKVStore):
    """MongoDB Key-Value store.

    Args:
        mongo_client (Any): MongoDB client
        uri (Optional[str]): MongoDB URI
        host (Optional[str]): MongoDB host
        port (Optional[int]): MongoDB port
        db_name (Optional[str]): MongoDB database name

    """

    def __init__(
        self,
        mongo_client: Any,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db_name: Optional[str] = None,
    ) -> None:
        """Init a MongoDBKVStore."""
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._client = cast(MongoClient, mongo_client)

        self._uri = uri
        self._host = host
        self._port = port

        self._db_name = db_name or "db_docstore"
        self._db = self._client[self._db_name]

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
    ) -> "MongoDBKVStore":
        """Load a MongoDBKVStore from a MongoDB URI.

        Args:
            uri (str): MongoDB URI
            db_name (Optional[str]): MongoDB database name

        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        mongo_client: MongoClient = MongoClient(uri)
        return cls(
            mongo_client=mongo_client,
            db_name=db_name,
            uri=uri,
        )

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
    ) -> "MongoDBKVStore":
        """Load a MongoDBKVStore from a MongoDB host and port.

        Args:
            host (str): MongoDB host
            port (int): MongoDB port
            db_name (Optional[str]): MongoDB database name

        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        mongo_client: MongoClient = MongoClient(host, port)
        return cls(
            mongo_client=mongo_client,
            db_name=db_name,
            host=host,
            port=port,
        )

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        val = val.copy()
        val["_id"] = key
        self._db[collection].replace_one(
            {"_id": key},
            val,
            upsert=True,
        )

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        result = self._db[collection].find_one({"_id": key})
        if result is not None:
            result.pop("_id")
            return result
        return None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store.

        Args:
            collection (str): collection name

        """
        results = self._db[collection].find()
        output = {}
        for result in results:
            key = result.pop("_id")
            output[key] = result
        return output

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        result = self._db[collection].delete_one({"_id": key})
        return result.deleted_count > 0
