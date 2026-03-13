import os

from contextlib import contextmanager
import random
import uuid
from types import SimpleNamespace
from typing import Iterator

import pytest

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters, VectorStoreQuery
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


@contextmanager
def _managed_vector_store(
    client: VectorAIClient, nodes_to_add: list[TextNode]
) -> Iterator[ActianVectorAIVectorStore]:
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    client.collections.create(
        collection_name,
        vectors_config=VectorParams(size=len(nodes_to_add[0].embedding), distance=Distance.Cosine),
    )
    vector_store = ActianVectorAIVectorStore(client, collection_name=collection_name)
    vector_store.add(nodes_to_add)
    assert vector_store._client.points.count(collection_name) == len(nodes_to_add)
    try:
        yield vector_store
    finally:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)


nodes = [
    TextNode(
        id_="c7ed938f-f8ef-4970-bf74-c240f33522f2",
        text="The 'uv' package manager provides a high-performance interface for Python dependency resolution.",
        embedding=get_mock_embedding(),
        metadata={"category": "tools", "score": 0.1, "tag": "alpha_token", "optional": "x"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_01")}
    ),
    TextNode(
        id_="0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
        text="PostgreSQL utilizes HugePages to reduce TLB miss overhead in high-concurrency environments.",
        embedding=get_mock_embedding(),
        metadata={"category": "database", "score": 0.5, "tag": "beta_token", "optional": "x"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_02")}
    ),
    TextNode(
        id_="2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
        text="LlamaIndex nodes are the atomic units of data used for building RAG applications.",
        embedding=get_mock_embedding(),
        metadata={"category": "ai", "score": 0.9, "tag": "gamma_token"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_03")}
    ),
    TextNode(
        id_="cd617e5a-fb20-4f6e-a616-259a711268bb",
        text="NMC batteries offer higher energy density compared to LFP cells but require careful thermal management.",
        embedding=get_mock_embedding(),
        metadata={"category": "energy", "score": 0.7, "tag": "unique_phrase_42", "optional": "y"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_03")}
    ),
    TextNode(
        id_="1ac88386-031d-4453-a636-b507365eb377",
        text="A 100-count cotton fabric provides a smoother texture suitable for high-end domestic bedding.",
        embedding=get_mock_embedding(),
        metadata={"category": "textile", "score": 0.3, "tag": "zeta_token"},
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
        assert info['title'] == "Actian VectorAI DB"

        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

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

        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

def test_clear() -> None:
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        with _managed_vector_store(client, nodes) as vector_store:
            assert client.collections.exists(vector_store._collection_name)

            vector_store.clear()

            assert not client.collections.exists(vector_store._collection_name)

@pytest.mark.parametrize(
    ("filters", "expected_remaining_count"),
    [
        (
            MetadataFilters(
                filters=[MetadataFilter(key="category", operator=FilterOperator.EQ, value="ai")]
            ),
            len(nodes) - 1,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.IN, value=["tools", "database"]
                    )
                ]
            ),
            len(nodes) - 2,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="category", operator=FilterOperator.EQ, value="ai"),
                    MetadataFilter(key="category", operator=FilterOperator.EQ, value="textile"),
                ],
                condition=FilterCondition.OR,
            ),
            len(nodes) - 2,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="category", operator=FilterOperator.EQ, value="ai"),
                    MetadataFilter(key="ref_doc_id", operator=FilterOperator.EQ, value="doc_03"),
                ],
                condition=FilterCondition.AND,
            ),
            len(nodes) - 1,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="ref_doc_id", operator=FilterOperator.EQ, value="doc_03"),
                    MetadataFilter(key="category", operator=FilterOperator.EQ, value="energy"),
                ],
                condition=FilterCondition.NOT,
            ),
            2,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category",
                        operator=FilterOperator.NIN,
                        value=["ai", "energy"],
                    )
                ]
            ),
            2,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="nonexistent"
                    )
                ]
            ),
            len(nodes),
        ),
    ],
)
def test_delete_nodes_with_metadata_filters(
    filters: MetadataFilters,
    expected_remaining_count: int,
) -> None:
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        with _managed_vector_store(client, nodes) as vector_store:
            vector_store.delete_nodes(filters=filters)

            assert (
                vector_store._client.points.count(vector_store._collection_name)
                == expected_remaining_count
            )


@pytest.mark.parametrize(
    ("operator", "key", "value", "expected_remaining_count", "raises_not_implemented"),
    [
        (FilterOperator.EQ, "category", "ai", 4, False),
        (FilterOperator.EQ, "score", 0.5, 4, False),
        (FilterOperator.GT, "score", 0.7, 4, False),
        (FilterOperator.LT, "score", 0.3, 4, False),
        (FilterOperator.NE, "score", 0.5, 1, False),
        (FilterOperator.NE, "category", "ai", 1, False),
        (FilterOperator.GTE, "score", 0.7, 3, False),
        (FilterOperator.LTE, "score", 0.3, 3, False),
        (FilterOperator.IN, "category", ["tools", "database"], 3, False),
        (FilterOperator.IN, "category", "tools,database", 3, False),
        (FilterOperator.NIN, "category", ["ai", "energy"], 2, False),
        (FilterOperator.NIN, "category", "ai,energy", 2, False),
        (FilterOperator.ANY, "tag", ["PHRASE"], 4, True),
        (FilterOperator.ALL, "tag", ["PHRASE"], 4, True),
        (FilterOperator.TEXT_MATCH, "tag", "phrase", 4, False),
        (FilterOperator.TEXT_MATCH_INSENSITIVE, "tag", "PHRASE", 4, True),
        (FilterOperator.CONTAINS, "tag", "unique_phrase_42", 4, True),
        (FilterOperator.IS_EMPTY, "optional", None, 3, False),
    ],
)
def test_delete_nodes_with_each_supported_filter_operator(
    operator: FilterOperator,
    key: str,
    value: object,
    expected_remaining_count: int,
    raises_not_implemented: bool,
) -> None:
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        with _managed_vector_store(client, nodes) as vector_store:

            if raises_not_implemented:
                with pytest.raises(NotImplementedError):
                    vector_store.delete_nodes(
                        filters=MetadataFilters(
                            filters=[MetadataFilter(key=key, operator=operator, value=value)]
                        )
                    )
            else:
                vector_store.delete_nodes(
                    filters=MetadataFilters(
                        filters=[MetadataFilter(key=key, operator=operator, value=value)]
                    )
                )

                assert (
                    vector_store._client.points.count(vector_store._collection_name)
                    == expected_remaining_count
                )
