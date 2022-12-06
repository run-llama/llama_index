"""Base schema for data structures."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List


from gpt_index.utils import get_new_id

@dataclass
class BaseDocument(ABC):
    """Base document.

    Generic abstract interfaces that captures both index structs
    as well as documents.

    """

    @property
    @abstractmethod
    def text(self) -> str:
        """Get text."""

    @property
    def doc_id(self) -> str:
        """Get doc_id."""
        raise NotImplementedError("Not implemented yet.")


@dataclass
class Document(BaseDocument):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    _text: str
    extra_info: Optional[Dict] = None
    # TODO: introduce concept of docstore for unique id's
    _doc_id: Optional[str] = None

    @property
    def text(self) -> str:
        """Get text."""
        return self._text

    @property
    def doc_id(self) -> str:
        """Get doc_id."""
        if self._doc_id is None:
            raise ValueError("doc_id not set.")
        return self._doc_id

    @text.setter
    def doc_id(self, doc_id: str) -> None:
        """Set doc_id."""
        self._doc_id = doc_id


@dataclass
class DocumentStore:
    """Document store."""

    docs: Dict[str, BaseDocument] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, docs: List[BaseDocument]) -> "DocumentStore":
        obj = cls()
        obj.add_documents(docs)
        return obj

    def get_new_id(self) -> str:
        """Get a new ID."""
        return get_new_id(set(self.docs.keys()))

    def add_documents(self, docs: List[BaseDocument], generate_id: bool = True) -> None:
        """Add a document to the store.

        If generate_id = True, then generate_id for doc if doesn't exist.
        
        """
        for doc in docs:
            if generate_id and doc.doc_id is None:
                doc.doc_id = self.get_new_id()
            self.docs[doc.doc_id] = doc

    def get_document(self, doc_id: str) -> Optional[BaseDocument]:
        """Get a document from the store."""
        return self.docs.get(doc_id, None)

    def __len__(self):
        return len(self.docs.keys())