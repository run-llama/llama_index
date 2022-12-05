"""Base embeddings file."""

from abc import abstractmethod
from typing import List

# TODO: change to numpy array
EMB_TYPE = List


class BaseEmbedding:
    """Base class for embeddings."""

    @abstractmethod
    def get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""

    @abstractmethod
    def get_text_embedding(self, query: str) -> List[float]:
        """Get text embedding."""

    @abstractmethod
    def similarity(self, embedding1: EMB_TYPE, embedding_2: EMB_TYPE) -> float:
        """Get embedding similarity."""
