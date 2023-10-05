from enum import Enum
from typing import Dict, Type

from llama_index.vector_stores.bagel import BagelVectorStore
from llama_index.vector_stores.cassandra import CassandraVectorStore
from llama_index.vector_stores.chatgpt_plugin import ChatGPTRetrievalPluginClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.vector_stores.epsilla import EpsillaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.myscale import MyScaleVectorStore
from llama_index.vector_stores.opensearch import OpensearchVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.rocksetdb import RocksetVectorStore
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.vector_stores.types import VectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore


class VectorStoreType(str, Enum):
    SIMPLE = "simple"
    REDIS = "redis"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    OPENSEARCH = "opensearch"
    FAISS = "faiss"
    CASSANDRA = "cassandra"
    CHROMA = "chroma"
    CHATGPT_PLUGIN = "chatgpt_plugin"
    LANCEDB = "lancedb"
    MILVUS = "milvus"
    DEEPLAKE = "deeplake"
    MYSCALE = "myscale"
    SUPABASE = "supabase"
    ROCKSET = "rockset"
    BAGEL = "bagel"
    EPSILLA = "epsilla"


VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS: Dict[VectorStoreType, Type[VectorStore]] = {
    VectorStoreType.SIMPLE: SimpleVectorStore,
    VectorStoreType.REDIS: RedisVectorStore,
    VectorStoreType.WEAVIATE: WeaviateVectorStore,
    VectorStoreType.QDRANT: QdrantVectorStore,
    VectorStoreType.LANCEDB: LanceDBVectorStore,
    VectorStoreType.SUPABASE: SupabaseVectorStore,
    VectorStoreType.MILVUS: MilvusVectorStore,
    VectorStoreType.PINECONE: PineconeVectorStore,
    VectorStoreType.OPENSEARCH: OpensearchVectorStore,
    VectorStoreType.FAISS: FaissVectorStore,
    VectorStoreType.CASSANDRA: CassandraVectorStore,
    VectorStoreType.CHROMA: ChromaVectorStore,
    VectorStoreType.CHATGPT_PLUGIN: ChatGPTRetrievalPluginClient,
    VectorStoreType.DEEPLAKE: DeepLakeVectorStore,
    VectorStoreType.MYSCALE: MyScaleVectorStore,
    VectorStoreType.ROCKSET: RocksetVectorStore,
    VectorStoreType.BAGEL: BagelVectorStore,
    VectorStoreType.EPSILLA: EpsillaVectorStore,
}

VECTOR_STORE_CLASS_TO_VECTOR_STORE_TYPE: Dict[Type[VectorStore], VectorStoreType] = {
    cls_: type_ for type_, cls_ in VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS.items()
}
