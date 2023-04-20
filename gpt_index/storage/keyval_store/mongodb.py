from typing import Any, Dict, List, Optional, cast
import uuid
from gpt_index.storage.keyval_store.types import BaseKeyValStore


IMPORT_ERROR_MSG = "`pymongo` package not found, please run `pip install pymongo`"


class MongoDBKeyValStore(BaseKeyValStore):
    def __init__(
        self,
        mongo_client: Any,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._client = cast(MongoClient, mongo_client)

        self._uri = uri
        self._host = host
        self._port = port

        self._db_name = db_name or "db_docstore"
        self._collection_name = collection_name or f"collection_{uuid.uuid4()}"
        self._hash_collection_name = f"{collection_name}/ref_doc_info"

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "MongoDBKeyValStore":
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        mongo_client: MongoClient = MongoClient(uri)
        return cls(
            mongo_client=mongo_client,
            db_name=db_name,
            collection_name=collection_name,
            uri=uri,
        )

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "MongoDBKeyValStore":
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        mongo_client: MongoClient = MongoClient(host, port)
        return cls(
            mongo_client=mongo_client,
            db_name=db_name,
            collection_name=collection_name,
            host=host,
            port=port,
        )

    @property
    def collection(self) -> Any:
        return self._client[self._db_name][self._collection_name]

    def add(self, key: str, val: dict) -> None:
        from pymongo.collection import Collection

        assert isinstance(self.collection, Collection)
        val["_id"] = key
        self.collection.replace_one(
            {"_id": key},
            val,
            upsert=True,
        )

    def get(self, key: str) -> Optional[dict]:
        result = self.collection.find_one({"_id": key})
        if result is not None:
            result.pop("_id")
            return result
        return None

    def get_all(self) -> Dict[str, dict]:
        results = self.collection.find()
        output = {}
        for result in results:
            key = result.pop("_id")
            output[key] = result
        return output

    def delete(self, key: str) -> bool:
        result = self.collection.delete_one({"key": key})
        return result.deleted_count > 0
