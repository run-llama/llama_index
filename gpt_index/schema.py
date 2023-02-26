"""Base schema for data structures."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from gpt_index.utils import get_new_id


@dataclass
class BaseDocument(DataClassJsonMixin):
    """Base document.

    Generic abstract interfaces that captures both index structs
    as well as documents.

    """

    # TODO: consolidate fields from Document/IndexStruct into base class
    text: Optional[str] = None
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None

    # extra fields
    extra_info: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Post init."""
        # assign doc_id if not set
        if self.doc_id is None:
            self.doc_id = get_new_id(set())

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Document type."""

    @classmethod
    def get_types(cls) -> List[str]:
        """Get Document type."""
        # TODO: remove this method
        # a hack to preserve backwards compatibility for vector indices
        return [cls.get_type()]

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

    def get_embedding(self) -> List[float]:
        """Get embedding.

        Errors if embedding is None.

        """
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding

    @property
    def extra_info_str(self) -> Optional[str]:
        """Extra info string."""
        if self.extra_info is None:
            return None

        return "\n".join([f"{k}: {str(v)}" for k, v in self.extra_info.items()])
