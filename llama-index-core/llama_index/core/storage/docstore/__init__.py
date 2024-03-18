# alias for backwards compatibility
from llama_index.core.storage.docstore.simple_docstore import (
    DocumentStore,
    SimpleDocumentStore,
)
from llama_index.core.storage.docstore.types import BaseDocumentStore

__all__ = [
    "BaseDocumentStore",
    "DocumentStore",
    "SimpleDocumentStore",
]
