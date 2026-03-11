import os

import random
from types import SimpleNamespace

import pytest

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQuery
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore

from actian_vectorai import (
    Distance, 
    VectorAIClient,
    VectorParams
)

VECTORAI_SERVER_URL = os.getenv("VECTORAI_SERVER_URL", "localhost:50051")

def get_mock_embedding():
    return [random.gauss(-1, 1) for _ in range(128)]
    
nodes = [
    TextNode(
        id_="c7ed938f-f8ef-4970-bf74-c240f33522f2",
        text="The 'uv' package manager provides a high-performance interface for Python dependency resolution.",
        embedding=get_mock_embedding(),
        metadata={"category": "tools"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_01")}
    ),
    TextNode(
        id_="0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
        text="PostgreSQL utilizes HugePages to reduce TLB miss overhead in high-concurrency environments.",
        embedding=get_mock_embedding(),
        metadata={"category": "database"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_02")}
    ),
    TextNode(
        id_="2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
        text="LlamaIndex nodes are the atomic units of data used for building RAG applications.",
        embedding=get_mock_embedding(),
        metadata={"category": "ai"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_03")}
    ),
    TextNode(
        id_="cd617e5a-fb20-4f6e-a616-259a711268bb",
        text="NMC batteries offer higher energy density compared to LFP cells but require careful thermal management.",
        embedding=get_mock_embedding(),
        metadata={"category": "energy"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_03")}
    ),
    TextNode(
        id_="1ac88386-031d-4453-a636-b507365eb377",
        text="A 100-count cotton fabric provides a smoother texture suitable for high-end domestic bedding.",
        embedding=get_mock_embedding(),
        metadata={"category": "textile"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_05")}
    )
]

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

def test_basic_add_and_delete() -> None:
    collection_name = "test_collection"
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        client.collections.create(
            collection_name,
            vectors_config=VectorParams(size=len(nodes[0].embedding), distance=Distance.Cosine),
        )

        vectorStore = ActianVectorAIVectorStore(client, collection_name=collection_name)

        print("Adding nodes to vector store...")

        vectorStore.add(nodes)

        assert vectorStore._client.points.count(collection_name) == len(nodes)

        print("Nodes added. Deleting one node...")

        vectorStore.delete("doc_03")

        assert vectorStore._client.points.count(collection_name) == len(nodes) - 2