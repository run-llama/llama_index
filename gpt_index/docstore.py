"""Document store."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from dataclasses_json import DataClassJsonMixin

from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.readers.schema.base import Document

DOC_TYPE = Union[IndexStruct, Document]

# type key: used to store type of document
TYPE_KEY = "__type__"


@dataclass
class DocumentStore(DataClassJsonMixin):
    """Document store."""

    docs: Dict[str, DOC_TYPE] = field(default_factory=dict)

    def serialize_to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        docs_dict = {}
        for doc_id, doc in self.docs.items():
            doc_dict = doc.to_dict()
            doc_dict[TYPE_KEY] = doc.get_type()
            docs_dict[doc_id] = doc_dict
        return {"docs": docs_dict}

    def contains_index_struct(self, exclude_ids: Optional[List[str]] = None) -> bool:
        """Check if contains index struct."""
        exclude_ids = exclude_ids or []
        for doc in self.docs.values():
            if isinstance(doc, IndexStruct) and doc.get_doc_id() not in exclude_ids:
                return True
        return False

    @classmethod
    def load_from_dict(
        cls,
        docs_dict: Dict[str, Any],
        type_to_struct: Optional[Dict[str, Type[IndexStruct]]] = None,
    ) -> "DocumentStore":
        """Load from dict."""
        docs_obj_dict = {}
        for doc_id, doc_dict in docs_dict["docs"].items():
            doc_type = doc_dict.pop(TYPE_KEY, None)
            if doc_type == "Document" or doc_type is None:
                doc: DOC_TYPE = Document.from_dict(doc_dict)
            else:
                if type_to_struct is None:
                    raise ValueError(
                        "type_to_struct must be provided if type is index struct."
                    )
                # try using IndexStructType to retrieve documents
                if doc_type not in type_to_struct:
                    raise ValueError(
                        f"doc_type {doc_type} not found in type_to_struct. "
                        "Make sure that it was registered in the index registry."
                    )
                doc = type_to_struct[doc_type].from_dict(doc_dict)
                # doc = index_struct_cls.from_dict(doc_dict)
            docs_obj_dict[doc_id] = doc
        return cls(docs=docs_obj_dict)

    @classmethod
    def from_documents(cls, docs: List[DOC_TYPE]) -> "DocumentStore":
        """Create from documents."""
        obj = cls()
        obj.add_documents(docs)
        return obj

    def update_docstore(self, other: "DocumentStore") -> None:
        """Update docstore."""
        self.docs.update(other.docs)

    def add_documents(self, docs: List[DOC_TYPE], allow_update: bool = False) -> None:
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

    def get_document(self, doc_id: str, raise_error: bool = True) -> Optional[DOC_TYPE]:
        """Get a document from the store."""
        doc = self.docs.get(doc_id, None)
        if doc is None and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")
        return doc

    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self.docs

    def delete_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[DOC_TYPE]:
        """Delete a document from the store."""
        doc = self.docs.pop(doc_id, None)
        if doc is None and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")
        return doc

    def __len__(self) -> int:
        """Get length."""
        return len(self.docs.keys())
