"""Base embeddings file."""

from abc import abstractmethod
from typing import List

import numpy as np

# TODO: change to numpy array
EMB_TYPE = List


class BaseEmbedding:
    """Base class for embeddings."""

    @abstractmethod
    def get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""

    @abstractmethod
    def get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""

    def similarity(self, embedding1: EMB_TYPE, embedding2: EMB_TYPE) -> float:
        """Get embedding similarity."""
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm
