"""Empty Index."""

from llama_index.indices.empty.base import EmptyIndex, GPTEmptyIndex
from llama_index.indices.empty.retrievers import EmptyIndexRetriever

__all__ = ["EmptyIndex", "EmptyIndexRetriever", "GPTEmptyIndex"]
