"""Query classes for vector store indices."""

from gpt_index.indices.query.vector_store.faiss import GPTFaissIndexQuery
from gpt_index.indices.query.vector_store.simple import GPTSimpleVectorIndexQuery

__all__ = ["GPTFaissIndexQuery", "GPTSimpleVectorIndexQuery"]
