"""Base schema for readers."""
from typing import Optional

from llama_index.bridge.langchain import Document as LCDocument

from llama_index.schema import TextNode


class Document(TextNode):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    _compat_fields = {"doc_id": "id_"}

    @classmethod
    def get_type(cls) -> str:
        """Get Document type."""
        return "Document"

    @property
    def doc_id(self) -> str:
        """Get document ID."""
        return self.id_

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._compat_fields:
            name = self._compat_fields[name]
        super().__setattr__(name, value)

    def to_langchain_format(self) -> LCDocument:
        """Convert struct to LangChain document format."""
        metadata = self.metadata or {}
        return LCDocument(page_content=self.text, metadata=metadata)

    @classmethod
    def from_langchain_format(cls, doc: LCDocument) -> "Document":
        """Convert struct from LangChain document format."""
        return cls(text=doc.page_content, metadata=doc.metadata)


class ImageDocument(Document):
    """Data document containing an image."""

    # base64 encoded image str
    image: Optional[str] = None
