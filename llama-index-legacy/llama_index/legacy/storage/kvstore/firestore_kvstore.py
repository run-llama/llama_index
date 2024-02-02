from typing import Any, Dict, List, Optional, Tuple

from llama_index.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

# keyword "_" is reserved in Firestore but referred in llama_index/constants.py.
FIELD_NAME_REPLACE_SET = {"__data__": "data", "__type__": "type"}
FIELD_NAME_REPLACE_GET = {"data": "__data__", "type": "__type__"}

# "/" is not supported in Firestore Collection ID.
SLASH_REPLACEMENT = "_"
IMPORT_ERROR_MSG = (
    "`firestore` package not found, please run `pip3 install google-cloud-firestore`"
)
USER_AGENT = "LlamaIndex"
DEFAULT_FIRESTORE_DATABASE = "(default)"


class FirestoreKVStore(BaseKVStore):
    """Firestore Key-Value store.

    Args:
        project (str): The project which the client acts on behalf of.
        database (str): The database name that the client targets.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        database: str = DEFAULT_FIRESTORE_DATABASE,
    ) -> None:
        try:
            from google.cloud.firestore_v1.async_client import AsyncClient
            from google.cloud.firestore_v1.client import Client
            from google.cloud.firestore_v1.services.firestore.transports.base import (
                DEFAULT_CLIENT_INFO,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        client_info = DEFAULT_CLIENT_INFO
        client_info.user_agent = USER_AGENT
        self._adb = AsyncClient(
            project=project, database=database, client_info=client_info
        )
        self._db = Client(project=project, database=database, client_info=client_info)

    def firestore_collection(self, collection: str) -> str:
        return collection.replace("/", SLASH_REPLACEMENT)

    def replace_field_name_set(self, val: Dict[str, Any]) -> Dict[str, Any]:
        val = val.copy()
        for k, v in FIELD_NAME_REPLACE_SET.items():
            if k in val:
                val[v] = val[k]
                val.pop(k)
        return val

    def replace_field_name_get(self, val: Dict[str, Any]) -> Dict[str, Any]:
        val = val.copy()
        for k, v in FIELD_NAME_REPLACE_GET.items():
            if k in val:
                val[v] = val[k]
                val.pop(k)
        return val

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Put a key-value pair into the Firestore collection.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name
        """
        collection_id = self.firestore_collection(collection)
        val = self.replace_field_name_set(val)
        doc = self._db.collection(collection_id).document(key)
        doc.set(val, merge=True)

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Put a key-value pair into the Firestore collection.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name
        """
        collection_id = self.firestore_collection(collection)
        val = self.replace_field_name_set(val)
        doc = self._adb.collection(collection_id).document(key)
        await doc.set(val, merge=True)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        batch = self._db.batch()
        for i, (key, val) in enumerate(kv_pairs, start=1):
            collection_id = self.firestore_collection(collection)
            val = self.replace_field_name_set(val)
            batch.set(self._db.collection(collection_id).document(key), val, merge=True)
            if i % batch_size == 0:
                batch.commit()
                batch = self._db.batch()
        batch.commit()

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Put a dictionary of key-value pairs into the Firestore collection.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            collection (str): collection name
        """
        batch = self._adb.batch()
        for i, (key, val) in enumerate(kv_pairs, start=1):
            collection_id = self.firestore_collection(collection)
            doc = self._adb.collection(collection_id).document(key)
            val = self.replace_field_name_set(val)
            batch.set(doc, val, merge=True)
            if i % batch_size == 0:
                await batch.commit()
                batch = self._adb.batch()
        await batch.commit()

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a key-value pair from the Firestore.

        Args:
            key (str): key
            collection (str): collection name
        """
        collection_id = self.firestore_collection(collection)
        result = self._db.collection(collection_id).document(key).get().to_dict()
        if not result:
            return None

        return self.replace_field_name_get(result)

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """Get a key-value pair from the Firestore.

        Args:
            key (str): key
            collection (str): collection name
        """
        collection_id = self.firestore_collection(collection)
        result = (
            await self._adb.collection(collection_id).document(key).get()
        ).to_dict()
        if not result:
            return None

        return self.replace_field_name_get(result)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the Firestore collection.

        Args:
            collection (str): collection name
        """
        collection_id = self.firestore_collection(collection)
        docs = self._db.collection(collection_id).list_documents()
        output = {}
        for doc in docs:
            key = doc.id
            val = self.replace_field_name_get(doc.get().to_dict())
            output[key] = val
        return output

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the Firestore collection.

        Args:
            collection (str): collection name
        """
        collection_id = self.firestore_collection(collection)
        docs = self._adb.collection(collection_id).list_documents()
        output = {}
        async for doc in docs:
            key = doc.id
            data = doc.get().to_dict()
            if data is None:
                continue
            val = self.replace_field_name_get(data)
            output[key] = val
        return output

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the Firestore.

        Args:
            key (str): key
            collection (str): collection name
        """
        collection_id = self.firestore_collection(collection)
        doc = self._db.collection(collection_id).document(key)
        doc.delete()
        return True

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the Firestore.

        Args:
            key (str): key
            collection (str): collection name
        """
        collection_id = self.firestore_collection(collection)
        doc = self._adb.collection(collection_id).document(key)
        await doc.delete()
        return True
