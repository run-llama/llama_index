"""Couchbase vector stores."""

from llama_index.vector_stores.couchbase.base import (
    CouchbaseVectorStore,  # Deprecated
    CouchbaseSearchVectorStore,  # FTS-based
    CouchbaseQueryVectorStore,  # GSI-based with BHIVE support
    CouchbaseVectorStoreBase,  # Base class
    QueryVectorSearchType,  # Enum for search types
)

__all__ = [
    "CouchbaseVectorStore",
    "CouchbaseSearchVectorStore",
    "CouchbaseQueryVectorStore",
    "CouchbaseVectorStoreBase",
    "QueryVectorSearchType",
]
