"""Vector-store based data structures."""

from gpt_index.indices.vector_store.base import GPTVectorStoreIndex
from gpt_index.indices.vector_store.retrievers import VectorIndexRetriever

__all__ = [
    "GPTVectorStoreIndex",
    "VectorIndexRetriever",
]
