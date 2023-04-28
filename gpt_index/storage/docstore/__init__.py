from gpt_index.storage.docstore.types import BaseDocumentStore
from gpt_index.storage.docstore.simple_docstore import SimpleDocumentStore
from gpt_index.storage.docstore.mongo_docstore import MongoDocumentStore
from gpt_index.storage.docstore.keyval_docstore import KVDocumentStore

# alias for backwards compatibility
from gpt_index.storage.docstore.simple_docstore import DocumentStore


__all__ = [
    "BaseDocumentStore",
    "DocumentStore",
    "SimpleDocumentStore",
    "MongoDocumentStore",
    "KVDocumentStore",
]
