"""Base data structure classes."""
from abc import abstractmethod
from typing import Any, Generic, List, Optional, TypeVar, cast

from gpt_index.indices.data_structs import IndexStruct
from gpt_index.schema import Document

IS = TypeVar("IS", bound=IndexStruct)

DEFAULT_MODE = "default"


class BaseGPTIndexQuery(Generic[IS]):
    """Base GPT Index Query.

    Helper class that is used to query an index. Can be called within `query`
    method of a BaseGPTIndex object, or instantiated independently.

    """

    def __init__(
        self,
        index_struct: IS,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None:
            raise ValueError("index_struct must be provided.")
        self._validate_index_struct(index_struct)
        self._index_struct = index_struct

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    def _validate_index_struct(self, index_struct: IS) -> None:
        """Validate the index struct."""
        pass

    @abstractmethod
    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""


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
    def _mode_to_query(self, mode: str, **query_kwargs: Any) -> BaseGPTIndexQuery:
        """Query mode to class."""

    def query(
        self,
        query_str: str,
        verbose: bool = False,
        mode: str = DEFAULT_MODE,
        **query_kwargs: Any
    ) -> str:
        """Answer a query."""
        query_obj = self._mode_to_query(mode, **query_kwargs)
        return query_obj.query(query_str, verbose=verbose)

    @classmethod
    @abstractmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "BaseGPTIndex":
        """Load from disk."""

    @abstractmethod
    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
