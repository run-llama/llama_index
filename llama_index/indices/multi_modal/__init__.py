"""Vector-store based data structures."""

from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.indices.multi_modal.retriever import MutliModalVectorIndexRetriever

__all__ = [
    "MultiModalVectorStoreIndex",
    "MutliModalVectorIndexRetriever",
]
