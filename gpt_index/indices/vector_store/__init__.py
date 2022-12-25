"""Vector-store based data structures."""

from gpt_index.indices.vector_store.faiss import GPTFaissIndex
from gpt_index.indices.vector_store.simple import GPTSimpleVectorIndex

__all__ = ["GPTFaissIndex", "GPTSimpleVectorIndex"]
