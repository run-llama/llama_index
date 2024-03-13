"""Vector stores."""

from llama_index.legacy.vector_stores.astra import AstraDBVectorStore
from llama_index.legacy.vector_stores.awadb import AwaDBVectorStore
from llama_index.legacy.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    CognitiveSearchVectorStore,
)
from llama_index.legacy.vector_stores.azurecosmosmongo import (
    AzureCosmosDBMongoDBVectorSearch,
)
from llama_index.legacy.vector_stores.bagel import BagelVectorStore
from llama_index.legacy.vector_stores.cassandra import CassandraVectorStore
from llama_index.legacy.vector_stores.chatgpt_plugin import ChatGPTRetrievalPluginClient
from llama_index.legacy.vector_stores.chroma import ChromaVectorStore
from llama_index.legacy.vector_stores.dashvector import DashVectorStore
from llama_index.legacy.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.legacy.vector_stores.docarray import (
    DocArrayHnswVectorStore,
    DocArrayInMemoryVectorStore,
)
from llama_index.legacy.vector_stores.elasticsearch import (
    ElasticsearchStore,
)
from llama_index.legacy.vector_stores.epsilla import EpsillaVectorStore
from llama_index.legacy.vector_stores.faiss import FaissVectorStore
from llama_index.legacy.vector_stores.lancedb import LanceDBVectorStore
from llama_index.legacy.vector_stores.lantern import LanternVectorStore
from llama_index.legacy.vector_stores.metal import MetalVectorStore
from llama_index.legacy.vector_stores.milvus import MilvusVectorStore
from llama_index.legacy.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.legacy.vector_stores.myscale import MyScaleVectorStore
from llama_index.legacy.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.legacy.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from llama_index.legacy.vector_stores.pgvecto_rs import PGVectoRsStore
from llama_index.legacy.vector_stores.pinecone import PineconeVectorStore
from llama_index.legacy.vector_stores.postgres import PGVectorStore
from llama_index.legacy.vector_stores.qdrant import QdrantVectorStore
from llama_index.legacy.vector_stores.redis import RedisVectorStore
from llama_index.legacy.vector_stores.rocksetdb import RocksetVectorStore
from llama_index.legacy.vector_stores.simple import SimpleVectorStore
from llama_index.legacy.vector_stores.singlestoredb import SingleStoreVectorStore
from llama_index.legacy.vector_stores.supabase import SupabaseVectorStore
from llama_index.legacy.vector_stores.tair import TairVectorStore
from llama_index.legacy.vector_stores.tencentvectordb import TencentVectorDB
from llama_index.legacy.vector_stores.timescalevector import TimescaleVectorStore
from llama_index.legacy.vector_stores.txtai import TxtaiVectorStore
from llama_index.legacy.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.legacy.vector_stores.upstash import UpstashVectorStore
from llama_index.legacy.vector_stores.weaviate import WeaviateVectorStore
from llama_index.legacy.vector_stores.zep import ZepVectorStore

__all__ = [
    "ElasticsearchStore",
    "SimpleVectorStore",
    "RedisVectorStore",
    "RocksetVectorStore",
    "FaissVectorStore",
    "TxtaiVectorStore",
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
    "PGVectoRsStore",
    "TimescaleVectorStore",
    "ZepVectorStore",
    "AwaDBVectorStore",
    "BagelVectorStore",
    "Neo4jVectorStore",
    "AzureAISearchVectorStore",
    "CognitiveSearchVectorStore",
    "EpsillaVectorStore",
    "SingleStoreVectorStore",
    "VectorStoreQuery",
    "VectorStoreQueryResult",
    "MetadataFilters",
    "MetadataFilter",
    "ExactMatchFilter",
    "FilterCondition",
    "FilterOperator",
    "DashVectorStore",
    "TencentVectorDB",
    "AstraDBVectorStore",
    "AzureCosmosDBMongoDBVectorSearch",
    "LanternVectorStore",
    "MongoDBAtlasVectorSearch",
    "UpstashVectorStore",
]
