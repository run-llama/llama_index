"""Document store."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from gpt_index.docstore.types import BaseDocumentStore
from gpt_index.docstore.utils import doc_to_json, json_to_doc
from gpt_index.schema import BaseDocument


@dataclass
class SimpleDocumentStore(BaseDocumentStore):
    """Document (Node) store.

    NOTE: at the moment, this store is primarily used to store Node objects.
    Each node will be assigned an ID.

    The same docstore can be reused across index structures. This
    allows you to reuse the same storage for multiple index structures;
    otherwise, each index would create a docstore under the hood.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

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
        docs: Optional[Dict[str, BaseDocument]] = None,
        ref_doc_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._docs = docs or {}
        self._ref_doc_info = ref_doc_info or defaultdict(dict)

    @property
    def docs(self) -> Dict[str, BaseDocument]:
        return self._docs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        docs_dict = {}
        for doc_id, doc in self.docs.items():
            docs_dict[doc_id] = doc_to_json(doc)
        return {"docs": docs_dict, "ref_doc_info": self._ref_doc_info}

    @classmethod
    def from_dict(
        cls,
        docs_dict: Dict[str, Any],
    ) -> "SimpleDocumentStore":
        """Load from dict.

        Args:
            docs_dict (Dict[str, Any]): dict of documents

        """
        docs_obj_dict = {
            doc_id: json_to_doc(doc_dict)
            for doc_id, doc_dict in docs_dict["docs"].items()
        }

        return cls(
            docs=docs_obj_dict,
            ref_doc_info=defaultdict(dict, **docs_dict.get("ref_doc_info", {})),
        )

    def update_docstore(self, other: "BaseDocumentStore") -> None:
        """Update docstore.

        Args:
            other (SimpleDocumentStore): docstore to update from

        """
        assert isinstance(other, SimpleDocumentStore)
        self.docs.update(other.docs)

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
            self.docs[doc.get_doc_id()] = doc
            self._ref_doc_info[doc.get_doc_id()]["doc_hash"] = doc.get_doc_hash()

    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        """Get a document from the store.

        Args:
            doc_id (str): document id
            raise_error (bool): raise error if doc_id not found

        """
        doc = self.docs.get(doc_id, None)
        if doc is None and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")
        return doc

    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self.docs

    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store."""
        doc = self.docs.pop(doc_id, None)
        self._ref_doc_info.pop(doc_id, None)
        if doc is None and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a given doc_id."""
        self._ref_doc_info[doc_id]["doc_hash"] = doc_hash

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the stored hash for a document, if it exists."""
        return self._ref_doc_info[doc_id].get("doc_hash", None)


# alias for backwards compatibility
DocumentStore = SimpleDocumentStore
