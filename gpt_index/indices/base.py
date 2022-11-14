"""Base data structure classes."""
from abc import abstractmethod
from typing import Dict, Set, List, Any, Generic, TypeVar

from gpt_index.schema import Document
from gpt_index.indices.data_structs import IndexStruct

IS = TypeVar("IS", bound=IndexStruct)


class BaseGPTIndex(Generic[IS]):
    """Base GPT Index."""

    def __init__(self, documents: List[Document]) -> None:
        """Initialize with parameters."""
        # build index struct in the init function
        self.documents = documents
        self.index_struct = self.build_index_from_documents(documents)

    @abstractmethod
    def build_index_from_documents(self, documents: List[Document]) -> IS:
        """Build the index from documents."""

    @abstractmethod
    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""

    @abstractmethod
    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "BaseGPTIndex":
        """Load from disk."""

    @abstractmethod
    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
