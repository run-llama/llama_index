from llama_index.storage.docstore.types import BaseDocumentStore
from llama_index.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.storage.docstore.mongo_docstore import MongoDocumentStore
from llama_index.storage.docstore.keyval_docstore import KVDocumentStore

# alias for backwards compatibility
from llama_index.storage.docstore.simple_docstore import DocumentStore


__all__ = [
    "BaseDocumentStore",
    "DocumentStore",
    "SimpleDocumentStore",
    "MongoDocumentStore",
    "KVDocumentStore",
]
