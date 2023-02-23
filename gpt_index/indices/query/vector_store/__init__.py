"""Query classes for vector store indices."""

from gpt_index.indices.query.vector_store.base import GPTVectorStoreIndexQuery
from gpt_index.indices.query.vector_store.chroma import GPTChromaIndexQuery

__all__ = ["GPTVectorStoreIndexQuery", "GPTChromeIndexQuery""]
