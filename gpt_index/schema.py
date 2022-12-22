"""Base schema for data structures."""
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from gpt_index.utils import get_new_id


@dataclass
class BaseDocument(ABC):
    """Base document.

    Generic abstract interfaces that captures both index structs
    as well as documents.

    """

    # TODO: consolidate fields from Document/IndexStruct into base class
    text: Optional[str] = None
    doc_id: Optional[str] = None

    def get_text(self) -> str:
        """Get text."""
        if self.text is None:
            raise ValueError("text field not set.")
        return self.text

    def get_doc_id(self) -> str:
        """Get doc_id."""
        if self.doc_id is None:
            raise ValueError("doc_id not set.")
        return self.doc_id

    @property
    def is_doc_id_none(self) -> bool:
        """Check if doc_id is None."""
        return self.doc_id is None


@dataclass
class DocumentStore(DataClassJsonMixin):
    """Document store."""

    docs: Dict[str, BaseDocument] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, docs: List[BaseDocument]) -> "DocumentStore":
        """Create from documents."""
        obj = cls()
        obj.add_documents(docs)
        return obj

    def get_new_id(self) -> str:
        """Get a new ID."""
        return get_new_id(set(self.docs.keys()))

    def update_docstore(self, other: "DocumentStore") -> None:
        """Update docstore."""
        self.docs.update(other.docs)

    def add_documents(self, docs: List[BaseDocument], generate_id: bool = True) -> None:
        """Add a document to the store.

        If generate_id = True, then generate id for doc if doc_id doesn't exist.

        """
        for doc in docs:
            if doc.is_doc_id_none:
                if generate_id:
                    doc.doc_id = self.get_new_id()
                else:
                    raise ValueError(
                        "doc_id not set (to generate id, please set generate_id=True)."
                    )

            # NOTE: doc could already exist in the store, but we overwrite it
            self.docs[doc.get_doc_id()] = doc

    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        """Get a document from the store."""
        doc = self.docs.get(doc_id, None)
        if doc is None and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")
        return doc

    def __len__(self) -> int:
        """Get length."""
        return len(self.docs.keys())
