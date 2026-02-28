"""
Built-Simple Research API Readers for LlamaIndex.

This package provides LlamaIndex readers for Built-Simple's research APIs:

- **BuiltSimplePubMedReader**: Search PubMed biomedical literature
- **BuiltSimpleArxivReader**: Search arXiv scientific preprints
- **BuiltSimpleWikipediaReader**: Search Wikipedia articles

All readers support semantic/hybrid search and return properly formatted
LlamaIndex Document objects with rich metadata.

Quick Start:
    >>> from llama_index.readers.builtsimple import BuiltSimplePubMedReader
    >>> reader = BuiltSimplePubMedReader()
    >>> docs = reader.load_data("cancer immunotherapy", limit=10)

For more information, visit: https://built-simple.ai
"""

from llama_index.readers.builtsimple.pubmed import BuiltSimplePubMedReader
from llama_index.readers.builtsimple.arxiv import BuiltSimpleArxivReader
from llama_index.readers.builtsimple.wikipedia import BuiltSimpleWikipediaReader
from llama_index.readers.builtsimple.base import BuiltSimpleAPIError

__all__ = [
    "BuiltSimplePubMedReader",
    "BuiltSimpleArxivReader",
    "BuiltSimpleWikipediaReader",
    "BuiltSimpleAPIError",
]

__version__ = "0.1.0"
