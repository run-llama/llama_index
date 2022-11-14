"""Base data structure classes."""
from abc import abstractmethod
from typing import Any, Generic, List, Optional, TypeVar, cast

from gpt_index.indices.data_structs import IndexStruct
from gpt_index.schema import Document

IS = TypeVar("IS", bound=IndexStruct)


class BaseGPTIndex(Generic[IS]):
    """Base GPT Index."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        index_struct: Optional[IS] = None,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and documents is None:
            raise ValueError("One of documents or index_struct must be provided.")
        if index_struct is not None and documents is not None:
            raise ValueError("Only one of documents or index_struct can be provided.")

        # build index struct in the init function
        if index_struct is not None:
            self._index_struct = index_struct
        else:
            documents = cast(List[Document], documents)
            self._index_struct = self.build_index_from_documents(documents)

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    @abstractmethod
    def build_index_from_documents(self, documents: List[Document]) -> IS:
        """Build the index from documents."""

    @abstractmethod
    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""

    @classmethod
    @abstractmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "BaseGPTIndex":
        """Load from disk."""

    @abstractmethod
    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
