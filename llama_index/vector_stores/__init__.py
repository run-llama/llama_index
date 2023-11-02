"""Vector stores."""


from llama_index.vector_stores.astra import AstraDBVectorStore
from llama_index.vector_stores.awadb import AwaDBVectorStore
from llama_index.vector_stores.bagel import BagelVectorStore
from llama_index.vector_stores.cassandra import CassandraVectorStore
from llama_index.vector_stores.chatgpt_plugin import ChatGPTRetrievalPluginClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.cogsearch import CognitiveSearchVectorStore
from llama_index.vector_stores.dashvector import DashVectorStore
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.vector_stores.docarray import (
    DocArrayHnswVectorStore,
    DocArrayInMemoryVectorStore,
)
from llama_index.vector_stores.elasticsearch import (
    ElasticsearchStore,
)
from llama_index.vector_stores.epsilla import EpsillaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.vector_stores.metal import MetalVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.myscale import MyScaleVectorStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.rocksetdb import RocksetVectorStore
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.singlestoredb import SingleStoreVectorStore
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.vector_stores.tair import TairVectorStore
from llama_index.vector_stores.tencentvectordb import TencentVectorDB
from llama_index.vector_stores.timescalevector import TimescaleVectorStore
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.weaviate import WeaviateVectorStore

__all__ = [
    "ElasticsearchStore",
    "SimpleVectorStore",
    "RedisVectorStore",
    "RocksetVectorStore",
    "FaissVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "QdrantVectorStore",
    "CassandraVectorStore",
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
    "SupabaseVectorStore",
    "PGVectorStore",
    "TimescaleVectorStore",
    "ZepVectorStore",
    "AwaDBVectorStore",
    "BagelVectorStore",
    "Neo4jVectorStore",
    "CognitiveSearchVectorStore",
    "EpsillaVectorStore",
    "SingleStoreVectorStore",
    "VectorStoreQuery",
    "VectorStoreQueryResult",
    "MetadataFilters",
    "DashVectorStore",
    "TencentVectorDB",
    "AstraDBVectorStore",
]
