"""Vector-store based data structures."""

from llama_index.indices.vector_store.base import VectorStoreIndex, GPTVectorStoreIndex
from llama_index.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
    VectorIndexRetriever,
)

__all__ = [
    "SentenceWindowVectorIndex",
    "SentenceWindowVectorRetriever",
    "VectorStoreIndex",
    "VectorIndexRetriever",
    "VectorIndexAutoRetriever",
    # legacy
    "GPTVectorStoreIndex",
]
