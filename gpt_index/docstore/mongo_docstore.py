from typing import Any, Dict, List, Optional, Sequence
import uuid
from gpt_index.data_structs.node_v2 import Node
from gpt_index.docstore.types import DocumentStore
from gpt_index.docstore.utils import doc_to_json, json_to_doc
from gpt_index.schema import BaseDocument

from pymongo import MongoClient
from pymongo.collection import Collection


class MongoDocumentStore(DocumentStore):
    def __init__(
        self,
        mongo_client: MongoClient,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self._client = mongo_client

        self._uri = uri
        self._host = host
        self._port = port

        self._db_name = db_name or "db_docstore"
        self._collection_name = collection_name or f"collection_{uuid.uuid4()}"

    @classmethod
    def from_uri(
        cls,
        uri: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "MongoDocumentStore":
        mongo_client = MongoClient(uri)
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
        port: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "MongoDocumentStore":
        mongo_client = MongoClient(host, port)
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
    def collection(self) -> Collection:
        return self._client[self._db_name][self._collection_name]

    @property
    def docs(self) -> Dict[str, BaseDocument]:
        results = self.collection.find()
        output = {}
        for result in results:
            result.pop("_id")
            doc_id = result["doc_id"]
            output[doc_id] = result
        return output

    def add_documents(
        self, docs: Sequence[BaseDocument], allow_update: bool = True
    ) -> List[str]:
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
        return [doc.doc_id for doc in docs]

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

    def delete_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        result = self.collection.delete_one({"doc_id": doc_id})
        if result.deleted_count == 0:
            if raise_error:
                raise ValueError(f"doc_id {doc_id} not found.")
            else:
                return None

    def get_nodes(self, node_ids: List[str], raise_error: bool = True) -> List[Node]:
        """Get nodes from docstore.

        Args:
            node_ids (List[str]): node ids
            raise_error (bool): raise error if node_id not found

        """
        return [self.get_node(node_id, raise_error=raise_error) for node_id in node_ids]

    def get_node(self, node_id: str, raise_error: bool = True) -> Node:
        """Get node from docstore.

        Args:
            node_id (str): node id
            raise_error (bool): raise error if node_id not found

        """
        doc = self.get_document(node_id, raise_error=raise_error)
        if not isinstance(doc, Node):
            raise ValueError(f"Document {node_id} is not a Node.")
        return doc

    def get_node_dict(self, node_id_dict: Dict[int, str]) -> Dict[int, Node]:
        """Get node dict from docstore given a mapping of index to node ids.

        Args:
            node_id_dict (Dict[int, str]): mapping of index to node ids

        """
        return {
            index: self.get_node(node_id) for index, node_id in node_id_dict.items()
        }
