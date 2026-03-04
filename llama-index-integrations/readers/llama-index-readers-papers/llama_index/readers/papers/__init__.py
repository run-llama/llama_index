"""Init file."""

from llama_index.readers.papers.arxiv.base import ArxivReader
from llama_index.readers.papers.pubmed.base import PubmedReader

__all__ = ["ArxivReader", "PubmedReader"]
