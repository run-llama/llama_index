"""Base embeddings file."""

from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from gpt_index.utils import globals_helper

# TODO: change to numpy array
EMB_TYPE = List


class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class BaseEmbedding:
    """Base class for embeddings."""

    def __init__(self) -> None:
        """Init params."""
        self._total_tokens_used = 0
        self._last_token_usage: Optional[int] = None
        self._tokenizer: Callable = globals_helper.tokenizer

    @abstractmethod
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""

    def get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query_embedding = self._get_query_embedding(query)
        query_tokens_count = len(self._tokenizer(query))
        self._total_tokens_used += query_tokens_count
        return query_embedding

    @abstractmethod
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""

    def get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        text_embedding = self._get_text_embedding(text)
        text_tokens_count = len(self._tokenizer(text))
        self._total_tokens_used += text_tokens_count
        return text_embedding

    def similarity(
        self,
        embedding1: EMB_TYPE,
        embedding2: EMB_TYPE,
        mode: SimilarityMode = SimilarityMode.DEFAULT,
    ) -> float:
        """Get embedding similarity."""
        if mode == SimilarityMode.EUCLIDEAN:
            return float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
        elif mode == SimilarityMode.DOT_PRODUCT:
            product = np.dot(embedding1, embedding2)
            return product
        else:
            product = np.dot(embedding1, embedding2)
            norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            return product / norm

    @property
    def total_tokens_used(self) -> int:
        """Get the total tokens used so far."""
        return self._total_tokens_used

    @property
    def last_token_usage(self) -> int:
        """Get the last token usage."""
        if self._last_token_usage is None:
            return 0
        return self._last_token_usage

    @last_token_usage.setter
    def last_token_usage(self, value: int) -> None:
        """Set the last token usage."""
        self._last_token_usage = value
