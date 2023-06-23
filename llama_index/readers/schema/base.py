"""Base schema for readers."""
from typing import Optional

from langchain.docstore.document import Document as LCDocument

from llama_index.schema import TextNode


class Document(TextNode):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    @classmethod
    def get_type(cls) -> str:
        """Get Document type."""
        return "Document"

    @property
    def doc_id(self) -> str:
        """Get document ID."""
        return self.node_id

    @doc_id.setter
    def doc_id(self, value: str) -> None:
        self._id = value

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
