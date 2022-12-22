"""Base reader class."""
from abc import abstractmethod
from typing import Any, List

from gpt_index.readers.schema.base import Document


class BaseReader:
    """Utilities for loading data from a directory."""

    @abstractmethod
    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
