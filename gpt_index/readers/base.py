"""Base reader class."""
from abc import abstractmethod
from gpt_index.schema import Document
from typing import List


class BaseReader:
    """Utilities for loading data from a directory."""

    @abstractmethod
    def load_data(self) -> List[str]:
        """Load data from the input directory."""
