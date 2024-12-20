"""Vector-store based data structures."""

from llama_index.core.indices.vector_store.base import (
    GPTVectorStoreIndex,
    VectorStoreIndex,
)
from llama_index.core.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
    VectorIndexRetriever,
)

__all__ = [
    "VectorStoreIndex",
    "VectorIndexRetriever",
    "VectorIndexAutoRetriever",
    # legacy
    "GPTVectorStoreIndex",
]
