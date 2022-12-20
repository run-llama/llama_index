"""Base reader class."""
from abc import abstractmethod
from typing import Any, List, Type

from gpt_index.data_structs import IndexStruct
from gpt_index.schema import Document


class BaseReader:
    """Utilities for loading data from a directory."""

    @abstractmethod
    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data."""

    def load_index(
        self, index_struct_cls: Type[IndexStruct], **load_kwargs: Any
    ) -> IndexStruct:
        """Load index struct."""
        raise NotImplementedError("load_index_struct not implemented.")

    def save_index(self, index_struct: IndexStruct, **save_kwargs: Any) -> None:
        """Save index struct."""
        raise NotImplementedError("save_index_struct not implemented.")
