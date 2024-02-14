from typing import Dict, Type

from llama_index.legacy.vector_stores.chroma import ChromaVectorStore
from llama_index.legacy.vector_stores.lantern import LanternVectorStore
from llama_index.legacy.vector_stores.pinecone import PineconeVectorStore
from llama_index.legacy.vector_stores.postgres import PGVectorStore
from llama_index.legacy.vector_stores.qdrant import QdrantVectorStore
from llama_index.legacy.vector_stores.types import BasePydanticVectorStore
from llama_index.legacy.vector_stores.weaviate import WeaviateVectorStore

LOADABLE_VECTOR_STORES: Dict[str, Type[BasePydanticVectorStore]] = {
    ChromaVectorStore.class_name(): ChromaVectorStore,
    QdrantVectorStore.class_name(): QdrantVectorStore,
    PineconeVectorStore.class_name(): PineconeVectorStore,
    PGVectorStore.class_name(): PGVectorStore,
    WeaviateVectorStore.class_name(): WeaviateVectorStore,
    LanternVectorStore.class_name(): LanternVectorStore,
}


def load_vector_store(data: dict) -> BasePydanticVectorStore:
    if isinstance(data, BasePydanticVectorStore):
        return data
    class_name = data.pop("class_name", None)
    if class_name is None:
        raise ValueError("class_name is required to load a vector store")

    if class_name not in LOADABLE_VECTOR_STORES:
        raise ValueError(f"Unable to load vector store of type {class_name}")

    # pop unused keys
    data.pop("flat_metadata", None)
    data.pop("stores_text", None)
    data.pop("is_embedding_query", None)

    if class_name == WeaviateVectorStore.class_name():
        import weaviate

        auth_config_dict = data.pop("auth_config", None)
        if auth_config_dict is not None:
            auth_config = None
            if "api_key" in auth_config_dict:
                auth_config = weaviate.AuthApiKey(**auth_config_dict)
            elif "username" in auth_config_dict:
                auth_config = weaviate.AuthClientPassword(**auth_config_dict)
            else:
                raise ValueError(
                    "Unable to load weaviate auth config, please use an auth "
                    "config with an api_key or username/password."
                )

            data["auth_config"] = auth_config

    return LOADABLE_VECTOR_STORES[class_name](**data)  # type: ignore
