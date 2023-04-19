from typing import Any, Dict, Optional, Sequence, cast
import uuid
from gpt_index.docstore.types import BaseDocumentStore
from gpt_index.docstore.utils import doc_to_json, json_to_doc
from gpt_index.schema import BaseDocument


IMPORT_ERROR_MSG = "`pymongo` package not found, please run `pip install pymongo`"


class MongoDocumentStore(BaseDocumentStore):
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
    ) -> "MongoDocumentStore":
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
    ) -> "MongoDocumentStore":
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MongoDocumentStore":
        if config_dict.get("mongo_client", None) is not None:
            return cls(**config_dict)
        elif config_dict.get("uri", None) is not None:
            return cls.from_uri(
                uri=config_dict["uri"],
                db_name=config_dict["db_name"],
                collection_name=config_dict["collection_name"],
            )
        elif (
            config_dict.get("host", None) is not None
            and config_dict.get("port", None) is not None
        ):
            return cls.from_host_and_port(
                host=config_dict["host"],
                port=config_dict["port"],
                db_name=config_dict["db_name"],
                collection_name=config_dict["collection_name"],
            )
        else:
            raise ValueError("Cannot construct MongoDocumentStore.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "db_name": self._db_name,
            "collection_name": self._collection_name,
            "uri": self._uri,
            "host": self._host,
            "port": self._port,
        }

    @property
    def client(self) -> Any:
        return self._client

    @property
    def collection(self) -> Any:
        return self._client[self._db_name][self._collection_name]

    @property
    def hash_collection(self) -> Any:
        return self._client[self._db_name][self._hash_collection_name]

    @property
    def docs(self) -> Dict[str, BaseDocument]:
        results = self.collection.find()
        output = {}
        for result in results:
            result.pop("_id")
            doc_id = result["doc_id"]
            output[doc_id] = json_to_doc(result)
        return output

    def add_documents(
        self, docs: Sequence[BaseDocument], allow_update: bool = True
    ) -> None:
        for doc in docs:
            if doc.is_doc_id_none:
                raise ValueError("doc_id not set")

            # NOTE: doc could already exist in the store, but we overwrite it
            if not allow_update and self.document_exists(doc.get_doc_id()):
                raise ValueError(
                    f"doc_id {doc.get_doc_id()} already exists. "
                    "Set allow_update to True to overwrite."
                )

            doc_json = doc_to_json(doc)
            self.collection.replace_one(
                {"doc_id": doc.doc_id},
                doc_json,
                upsert=True,
            )

    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        result = self.collection.find_one({"doc_id": doc_id})
        return result is not None

    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        result = self.collection.find_one({"doc_id": doc_id})
        if result is None:
            if raise_error:
                raise ValueError(f"doc_id {doc_id} not found.")
            else:
                return None
        result.pop("_id")
        return json_to_doc(result)

    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        result = self.collection.delete_one({"doc_id": doc_id})
        if result.deleted_count == 0:
            if raise_error:
                raise ValueError(f"doc_id {doc_id} not found.")
            else:
                return None

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a given doc_id."""
        self.hash_collection.replace_one(
            {"doc_id": doc_id},
            {
                "doc_id": doc_id,
                "doc_hash": doc_hash,
            },
            upsert=True,
        )

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the stored hash for a document, if it exists."""
        obj = self.hash_collection.find_one(filter={"doc_id": doc_id})
        if obj is not None:
            return obj.get("doc_hash", None)
        return None

    def update_docstore(self, other: "BaseDocumentStore") -> None:
        """Update docstore.

        Args:
            other (BaseDocumentStore): docstore to update from

        """
        self.add_documents(list(other.docs.values()))
