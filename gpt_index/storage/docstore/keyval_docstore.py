"""Document store."""

from typing import Dict, Optional, Sequence
import uuid

from gpt_index.storage.docstore.types import BaseDocumentStore
from gpt_index.storage.docstore.utils import doc_to_json, json_to_doc
from gpt_index.schema import BaseDocument
from gpt_index.storage.kvstore.types import (
    BaseKVStore,
)


class KeyValDocumentStore(BaseDocumentStore):
    """Document (Node) store.

    NOTE: at the moment, this store is primarily used to store Node objects.
    Each node will be assigned an ID.

    The same docstore can be reused across index structures. This
    allows you to reuse the same storage for multiple index structures;
    otherwise, each index would create a docstore under the hood.

    .. code-block:: python
        nodes = SimpleNodeParser.get_nodes_from_documents()
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)

        list_index = GPTListIndex(nodes, docstore=docstore)
        vector_index = GPTSimpleVectorIndex(nodes, docstore=docstore)
        keyword_table_index = GPTSimpleKeywordTableIndex(nodes, docstore=docstore)

    This will use the same docstore for multiple index structures.

    Args:
        docs (Dict[str, BaseDocument]): documents
        ref_doc_info (Dict[str, Dict[str, Any]]): reference document info

    """

    def __init__(
        self,
        keyval_store: BaseKVStore,
        namespace: Optional[str] = None,
    ):
        self._keyval_store = keyval_store
        namespace = namespace or str(uuid.uuid4())
        self._collection = f"{namespace}/data"
        self._metadata_collection = f"{namespace}/metadata"

    @property
    def docs(self) -> Dict[str, BaseDocument]:
        json_dict = self._keyval_store.get_all(collection=self._collection)
        return {key: json_to_doc(json) for key, json in json_dict.items()}

    def add_documents(
        self, docs: Sequence[BaseDocument], allow_update: bool = True
    ) -> None:
        """Add a document to the store.

        Args:
            docs (List[BaseDocument]): documents
            allow_update (bool): allow update of docstore from document

        """
        for doc in docs:
            if doc.is_doc_id_none:
                raise ValueError("doc_id not set")

            # NOTE: doc could already exist in the store, but we overwrite it
            if not allow_update and self.document_exists(doc.get_doc_id()):
                raise ValueError(
                    f"doc_id {doc.get_doc_id()} already exists. "
                    "Set allow_update to True to overwrite."
                )
            key = doc.get_doc_id()
            data = doc_to_json(doc)
            metadata = {"doc_hash": doc.get_doc_hash()}
            self._keyval_store.put(key, data, collection=self._collection)
            self._keyval_store.put(key, metadata, collection=self._metadata_collection)

    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        """Get a document from the store.

        Args:
            doc_id (str): document id
            raise_error (bool): raise error if doc_id not found

        """
        json = self._keyval_store.get(doc_id, collection=self._collection)
        if json is None:
            if raise_error:
                raise ValueError(f"doc_id {doc_id} not found.")
            else:
                return None
        return json_to_doc(json)

    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        return self._keyval_store.get(doc_id, self._collection) is not None

    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store."""
        delete_success = self._keyval_store.delete(doc_id, collection=self._collection)
        _ = self._keyval_store.delete(doc_id, collection=self._metadata_collection)
        if not delete_success and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a given doc_id."""
        metadata = {"doc_hash": doc_hash}
        self._keyval_store.put(doc_id, metadata, collection=self._metadata_collection)

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the stored hash for a document, if it exists."""
        metadata = self._keyval_store.get(doc_id, collection=self._metadata_collection)
        if metadata is not None:
            return metadata.get("doc_hash", None)
        else:
            return None
