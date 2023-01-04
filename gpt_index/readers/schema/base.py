"""Base schema for readers."""
from dataclasses import dataclass
from typing import Dict, Optional

from gpt_index.schema import BaseDocument
from langchain.docstore.document import Document as LCDocument


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

    def to_langchain_format(self):
        """Convert struct to LangChain document format."""
        metadata = self.extra_info or {}
        return LCDocument(page_content=self.text, metadata=metadata)