"""Base schema for data structures."""
from abc import abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from llama_index.utils import get_new_id


def _validate_is_flat_dict(metadata_dict: dict) -> None:
    """
    Validate that metadata dict is flat,
    and key is str, and value is one of (str, int, float).
    """
    for key, val in metadata_dict.items():
        if not isinstance(key, str):
            raise ValueError("Metadata key must be str!")
        if not isinstance(val, (str, int, float)):
            raise ValueError("Value must be one of (str, int, float)")


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
    doc_hash: Optional[str] = None

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - used by vector DBs for metadata filtering

    This must be a flat dictionary, 
    and only uses str keys, and (str, int, float) values.
    """
    extra_info: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Post init."""
        # assign doc_id if not set
        if self.doc_id is None:
            self.doc_id = get_new_id(set())
        if self.doc_hash is None:
            self.doc_hash = self._generate_doc_hash()

        if self.extra_info is not None:
            _validate_is_flat_dict(self.extra_info)

    def _generate_doc_hash(self) -> str:
        """Generate a hash to represent the document."""
        doc_identity = str(self.text) + str(self.extra_info)
        return sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest()

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

    def get_doc_hash(self) -> str:
        """Get doc_hash."""
        if self.doc_hash is None:
            raise ValueError("doc_hash is not set.")
        return self.doc_hash

    @property
    def is_doc_id_none(self) -> bool:
        """Check if doc_id is None."""
        return self.doc_id is None

    @property
    def is_text_none(self) -> bool:
        """Check if text is None."""
        return self.text is None

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
