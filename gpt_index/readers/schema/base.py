"""Base schema for readers."""
from dataclasses import dataclass
from typing import Dict, Optional, List

from gpt_index.schema import BaseDocument


@dataclass
class Document(BaseDocument):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    embedding: Optional[List[float]] = None
    extra_info: Optional[Dict] = None

    def __post_init__(self) -> None:
        """Post init."""
        if self.text is None:
            raise ValueError("text field not set.")

    def set_embedding(self, embedding: List[float]) -> None:
        """Set embedding."""
        self.embedding = embedding

    def get_embedding(self) -> List[float]:
        """Get embedding."""
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding