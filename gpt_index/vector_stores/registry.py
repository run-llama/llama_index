from enum import Enum
from typing import Any, Dict, Optional, Type
from gpt_index.constants import DATA_KEY, TYPE_KEY
from gpt_index.vector_stores.chatgpt_plugin import ChatGPTRetrievalPluginClient
from gpt_index.vector_stores.chroma import ChromaVectorStore
from gpt_index.vector_stores.faiss import FaissVectorStore
from gpt_index.vector_stores.opensearch import OpensearchVectorStore
from gpt_index.vector_stores.pinecone import PineconeVectorStore
from gpt_index.vector_stores.qdrant import QdrantVectorStore
from gpt_index.vector_stores.simple import SimpleVectorStore

from gpt_index.vector_stores.types import VectorStore
from gpt_index.vector_stores.weaviate import WeaviateVectorStore


class VectorStoreType(str, Enum):
    SIMPLE = "simple"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    OPENSEARCH = "opensearch"
    FAISS = "faiss"
    CHROMA = "chroma"
    CHATGPT_PLUGIN = "chatgpt_plugin"


VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS: Dict[VectorStoreType, Type[VectorStore]] = {
    VectorStoreType.SIMPLE: SimpleVectorStore,
    VectorStoreType.WEAVIATE: WeaviateVectorStore,
    VectorStoreType.QDRANT: QdrantVectorStore,
    VectorStoreType.PINECONE: PineconeVectorStore,
    VectorStoreType.OPENSEARCH: OpensearchVectorStore,
    VectorStoreType.FAISS: FaissVectorStore,
    VectorStoreType.CHROMA: ChromaVectorStore,
    VectorStoreType.CHATGPT_PLUGIN: ChatGPTRetrievalPluginClient,
}

VECTOR_STORE_CLASS_TO_VECTOR_STORE_TYPE: Dict[Type[VectorStore], VectorStoreType] = {
    cls_: type_ for type_, cls_ in VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS.items()
}


def load_vector_store_from_dict(
    vector_store_dict: Dict[str, Any],
    type_to_cls: Optional[Dict[VectorStoreType, Type[VectorStore]]] = None,
    **kwargs: Any,
) -> VectorStore:
    type_to_cls = type_to_cls or VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS
    type = vector_store_dict[TYPE_KEY]
    config_dict: Dict[str, Any] = vector_store_dict[DATA_KEY]

    # Inject kwargs into data dict.
    # This allows us to explicitly pass in unserializable objects
    # like the vector store client.
    config_dict.update(kwargs)

    cls = type_to_cls[type]
    return cls.from_dict(config_dict)


def save_vector_store_to_dict(
    vector_store: VectorStore,
    cls_to_type: Optional[Dict[Type[VectorStore], VectorStoreType]] = None,
) -> Dict[str, Any]:
    cls_to_type = cls_to_type or VECTOR_STORE_CLASS_TO_VECTOR_STORE_TYPE
    type_ = cls_to_type[type(vector_store)]
    return {TYPE_KEY: type_, DATA_KEY: vector_store.config_dict}
