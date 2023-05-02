"""Empty Index."""

from gpt_index.indices.empty.base import GPTEmptyIndex
from gpt_index.indices.empty.retrievers import EmptyIndexRetriever

__all__ = ["GPTEmptyIndex", "EmptyIndexRetriever"]
