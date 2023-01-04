"""Base schema for readers."""
from dataclasses import dataclass
from typing import Dict, Optional

from langchain.docstore.document import Document as LCDocument

from gpt_index.schema import BaseDocument


@dataclass
class Document(BaseDocument):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    extra_info: Optional[Dict] = None

    def __post_init__(self) -> None:
        """Post init."""
        if self.text is None:
            raise ValueError("text field not set.")

    def to_langchain_format(self) -> LCDocument:
        """Convert struct to LangChain document format."""
        metadata = self.extra_info or {}
        return LCDocument(page_content=self.text, metadata=metadata)

    @classmethod
    def from_langchain_format(cls, doc: LCDocument) -> "Document":
        """Convert struct from LangChain document format."""
        return cls(text=doc.page_content, extra_info=doc.metadata)
