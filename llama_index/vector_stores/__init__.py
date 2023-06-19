"""Vector stores."""
import warnings

from llama_index.vector_stores.chatgpt_plugin import ChatGPTRetrievalPluginClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.vector_stores.docarray import (
    DocArrayHnswVectorStore,
    DocArrayInMemoryVectorStore,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.vector_stores.metal import MetalVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.myscale import MyScaleVectorStore
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from llama_index.vector_stores.pgvector import PgVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.tair import TairVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore


# Backwards compatibility. Warn about deprecation.
class SupabaseVectorStore(PgVectorStore):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SupabaseVectorStore is deprecated, please use PgVectorStore instead",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "SimpleVectorStore",
    "RedisVectorStore",
    "FaissVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "MetalVectorStore",
    "OpensearchVectorStore",
    "OpensearchVectorClient",
    "ChatGPTRetrievalPluginClient",
    "MilvusVectorStore",
    "DeepLakeVectorStore",
    "MyScaleVectorStore",
    "LanceDBVectorStore",
    "TairVectorStore",
    "DocArrayInMemoryVectorStore",
    "DocArrayHnswVectorStore",
    "PgVectorStore",
    "SupabaseVectorStore",
]
