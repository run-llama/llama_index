"""Base schema for data structures."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from gpt_index.utils import get_new_id


@dataclass
class Document:
    """Generic interface for document."""

    doc_id: str
    text: str
    extra_info: Optional[Dict] = None


@dataclass
class DocumentStore:
    """Document store."""

    docs: Dict[str, Document] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, docs: List[Document]) -> "DocumentStore":
        obj = cls()
        obj.add_documents(docs)
        return obj

    def get_new_id(self) -> str:
        """Get a new ID."""
        return get_new_id(set(self.docs.keys()))

    def add_documents(self, docs: List[Document]) -> None:
        """Add a document to the store."""
        for doc in docs:
            self.docs[doc.doc_id] = doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document from the store."""
        return self.docs.get(doc_id, None)

    def __len__(self):
        return len(self.docs.keys())
