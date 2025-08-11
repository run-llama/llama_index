"""
Superlinked retriever integration for LlamaIndex.

Preferred import path:

    from llama_index.retrievers.superlinked import SuperlinkedRetriever

This module re-exports the implementation from the package-local namespace
to match the conventions of other LlamaIndex integration packages.
"""

from llama_index_retrievers_superlinked import SuperlinkedRetriever

__all__ = ["SuperlinkedRetriever"]
