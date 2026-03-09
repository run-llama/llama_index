import os

from types import SimpleNamespace

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQuery
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore

from actian_vectorai import (
    Distance, 
    VectorAIClient,
    VectorParams
)

VECTORAI_SERVER_URL = os.getenv("VECTORAI_SERVER_URL", "localhost:50051")

def test_class_name() -> None:
    assert ActianVectorAIVectorStore.class_name() == "ActianVectorAIVectorStore"

def test_init_vector_store() -> None:
    collection_name = "test_collection"
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        client.collections.create(
            collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.Cosine),
        )

        vectorStore = ActianVectorAIVectorStore(client, collection_name=collection_name)
        assert vectorStore._client is client
        assert vectorStore._collection_name == collection_name
        info = vectorStore._client.health_check()
        assert info['title'] == "VDSS"

def test_init_failed_unconnected_client() -> None:
    collection_name = "test_collection"
    client = VectorAIClient(VECTORAI_SERVER_URL)
    with pytest.raises(ValueError, match="ActianVectorAIVectorStore requires a connected VectorAIClient."):
        ActianVectorAIVectorStore(client, collection_name=collection_name)

def test_init_failed_nonexistent_collection() -> None:
    collection_name = "nonexistent_collection"
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        with pytest.raises(ValueError, match=f"Collection '{collection_name}' does not exist in Actian Vector AI."):
            ActianVectorAIVectorStore(client, collection_name=collection_name)