"""Document store."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from dataclasses_json import DataClassJsonMixin

from gpt_index.constants import TYPE_KEY
from gpt_index.data_structs.node_v2 import ImageNode, IndexNode, Node
from gpt_index.readers.schema.base import Document
from gpt_index.schema import BaseDocument


@dataclass
class DocumentStore(DataClassJsonMixin):
    """Document store."""

    docs: Dict[str, BaseDocument] = field(default_factory=dict)
    ref_doc_info: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def serialize_to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        docs_dict = {}
        for doc_id, doc in self.docs.items():
            doc_dict = doc.to_dict()
            doc_dict[TYPE_KEY] = doc.get_type()
            docs_dict[doc_id] = doc_dict
        return {"docs": docs_dict, "ref_doc_info": self.ref_doc_info}

    @classmethod
    def load_from_dict(
        cls,
        docs_dict: Dict[str, Any],
    ) -> "DocumentStore":
        """Load from dict."""
        docs_obj_dict = {}
        for doc_id, doc_dict in docs_dict["docs"].items():
            doc_type = doc_dict.pop(TYPE_KEY, None)
            doc: BaseDocument
            if doc_type == "Document" or doc_type is None:
                doc = Document.from_dict(doc_dict)
            elif doc_type == Node.get_type():
                doc = Node.from_dict(doc_dict)
            elif doc_type == ImageNode.get_type():
                doc = ImageNode.from_dict(doc_dict)
            elif doc_type == IndexNode.get_type():
                doc = IndexNode.from_dict(doc_dict)
            else:
                raise ValueError(f"Unknown doc type: {doc_type}")

            docs_obj_dict[doc_id] = doc
        return cls(
            docs=docs_obj_dict,
            ref_doc_info=defaultdict(dict, **docs_dict.get("ref_doc_info", {})),
        )

    @classmethod
    def from_documents(cls, docs: Sequence[BaseDocument]) -> "DocumentStore":
        """Create from documents."""
        obj = cls()
        obj.add_documents(docs)
        return obj

    @classmethod
    def merge(cls, docstores: Sequence["DocumentStore"]) -> "DocumentStore":
        merged_docstore = cls()
        for docstore in docstores:
            merged_docstore.update_docstore(docstore)
        return merged_docstore

    def update_docstore(self, other: "DocumentStore") -> None:
        """Update docstore."""
        self.docs.update(other.docs)

    def add_documents(
        self, docs: Sequence[BaseDocument], allow_update: bool = False
    ) -> None:
        """Add a document to the store."""
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
            self.ref_doc_info[doc.get_doc_id()]["doc_hash"] = doc.get_doc_hash()

    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        """Get a document from the store."""
        doc = self.docs.get(doc_id, None)
        if doc is None and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")
        return doc

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a given doc_id."""
        self.ref_doc_info[doc_id]["doc_hash"] = doc_hash

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the stored hash for a document, if it exists."""
        return self.ref_doc_info[doc_id].get("doc_hash", None)

    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self.docs

    def delete_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        """Delete a document from the store."""
        doc = self.docs.pop(doc_id, None)
        self.ref_doc_info.pop(doc_id, None)
        if doc is None and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")
        return doc

    def get_nodes(self, node_ids: List[str], raise_error: bool = True) -> List[Node]:
        """Get nodes from docstore."""
        return [self.get_node(node_id, raise_error=raise_error) for node_id in node_ids]

    def get_node(self, node_id: str, raise_error: bool = True) -> Node:
        """Get node from docstore."""
        doc = self.get_document(node_id, raise_error=raise_error)
        if not isinstance(doc, Node):
            raise ValueError(f"Document {node_id} is not a Node.")
        return doc

    def get_node_dict(self, node_id_dict: Dict[int, str]) -> Dict[int, Node]:
        """Get node dict from docstore given a mapping of index to node ids."""
        return {
            index: self.get_node(node_id) for index, node_id in node_id_dict.items()
        }
