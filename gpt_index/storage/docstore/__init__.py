from gpt_index.docstore.types import BaseDocumentStore
from gpt_index.docstore.simple_docstore import SimpleDocumentStore

# alias for backwards compatibility
from gpt_index.docstore.simple_docstore import DocumentStore
from gpt_index.docstore.mongo_docstore import MongoDocumentStore


__all__ = [
    "BaseDocumentStore",
    "SimpleDocumentStore",
    "DocumentStore",
    "MongoDocumentStore",
]
