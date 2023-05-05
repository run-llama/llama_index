"""Vector-store based data structures."""

from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever

__all__ = [
    "GPTVectorStoreIndex",
    "VectorIndexRetriever",
]
