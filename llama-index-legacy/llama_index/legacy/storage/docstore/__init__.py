from llama_index.legacy.storage.docstore.dynamodb_docstore import DynamoDBDocumentStore
from llama_index.legacy.storage.docstore.firestore_docstore import (
    FirestoreDocumentStore,
)
from llama_index.legacy.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.legacy.storage.docstore.mongo_docstore import MongoDocumentStore
from llama_index.legacy.storage.docstore.redis_docstore import RedisDocumentStore

# alias for backwards compatibility
from llama_index.legacy.storage.docstore.simple_docstore import (
    DocumentStore,
    SimpleDocumentStore,
)
from llama_index.legacy.storage.docstore.types import BaseDocumentStore

__all__ = [
    "BaseDocumentStore",
    "DocumentStore",
    "FirestoreDocumentStore",
    "SimpleDocumentStore",
    "MongoDocumentStore",
    "KVDocumentStore",
    "RedisDocumentStore",
    "DynamoDBDocumentStore",
]
