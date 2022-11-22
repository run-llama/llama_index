"""Base reader class."""
from abc import abstractmethod
from typing import List, Any

from gpt_index.schema import Document


class BaseReader:
    """Utilities for loading data from a directory."""

    @abstractmethod
    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
