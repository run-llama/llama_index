"""Base schema for data structures."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


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

    @property
    def text(self) -> str:
        """Get text."""
        return self._text
