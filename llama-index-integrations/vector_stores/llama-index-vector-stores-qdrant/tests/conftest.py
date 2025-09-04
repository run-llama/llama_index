import os
from typing import AsyncGenerator

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import pytest_asyncio
from qdrant_client.http import models


@pytest_asyncio.fixture
async def vector_store() -> QdrantVectorStore:
    client = qdrant_client.QdrantClient(":memory:")
    aclient = qdrant_client.AsyncQdrantClient(":memory:")
    vector_store = QdrantVectorStore("test", client=client, aclient=aclient)

    nodes = [
        TextNode(
            text="test1",
            id_="11111111-1111-1111-1111-111111111111",
            embedding=[1.0, 0.0],
            metadata={"some_key": 1},
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
        ),
        TextNode(
            text="test2",
            id_="22222222-2222-2222-2222-222222222222",
            embedding=[0.0, 1.0],
            metadata={"some_key": 2},
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
        ),
        TextNode(
            text="test3",
            id_="33333333-3333-3333-3333-333333333333",
            embedding=[1.0, 1.0],
            metadata={"some_key": "3"},
        ),
    ]

    vector_store.add(nodes)

    # in-memory client does not share data between instances
    await vector_store.async_add(nodes)

    return vector_store


@pytest_asyncio.fixture
async def hybrid_vector_store() -> QdrantVectorStore:
    client = qdrant_client.QdrantClient(":memory:")
    aclient = qdrant_client.AsyncQdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        "test",
        client=client,
        aclient=aclient,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )

    nodes = [
        TextNode(
            text="test1",
            id_="11111111-1111-1111-1111-111111111111",
            embedding=[1.0, 0.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
        ),
        TextNode(
            text="test2",
            id_="22222222-2222-2222-2222-222222222222",
            embedding=[0.0, 1.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
        ),
        TextNode(
            text="test3",
            id_="33333333-3333-3333-3333-333333333333",
            embedding=[1.0, 1.0],
        ),
    ]

    vector_store.add(nodes)

    # in-memory client does not share data between instances
    await vector_store.async_add(nodes)

    return vector_store


@pytest_asyncio.fixture
async def shard_vector_store() -> AsyncGenerator[QdrantVectorStore, None]:
    """
    Build a QdrantVectorStore configured for *custom sharding* and seed it with three nodes.

    Important:
      - This should connect to a Qdrant distributed instance (not ':memory:').
      - Ensure that the Qdrant instance is properly set up and accessible.

    """
    url = os.getenv("QDRANT_CLUSTER_URL")
    assert url, "QDRANT_CLUSTER_URL must be set for sharding tests"

    client = qdrant_client.QdrantClient(url)
    aclient = qdrant_client.AsyncQdrantClient(url)
    collection_name = "test"

    vector_store = QdrantVectorStore(
        collection_name,
        client=client,
        aclient=aclient,
        sharding_method=models.ShardingMethod.CUSTOM,
        shard_key_selector_fn=lambda x: x % 3,
        shard_keys=[0, 1, 2],
    )

    node_1 = TextNode(
        text="test1",
        id_="11111111-1111-1111-1111-111111111111",
        embedding=[1.0, 0.0],
        metadata={"some_key": 1},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
    )

    vector_store.add([node_1], shard_identifier=1)

    # in-memory client does not share data between instances
    await vector_store.async_add([node_1], shard_identifier=1)

    node_2 = TextNode(
        text="test2",
        id_="22222222-2222-2222-2222-222222222222",
        embedding=[0.0, 1.0],
        metadata={"some_key": 2},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
    )

    vector_store.add([node_2], shard_identifier=2)

    # in-memory client does not share data between instances
    await vector_store.async_add([node_2], shard_identifier=2)

    node_3 = TextNode(
        text="test3",
        id_="33333333-3333-3333-3333-333333333333",
        embedding=[1.0, 1.0],
        metadata={"some_key": "3"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
    )

    vector_store.add([node_3], shard_identifier=3)

    # in-memory client does not share data between instances
    await vector_store.async_add([node_3], shard_identifier=3)

    try:
        yield vector_store
    finally:
        try:
            await aclient.delete_collection(collection_name=collection_name)
        except Exception:
            pass
        try:
            await aclient.close()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass


@pytest_asyncio.fixture
async def payload_indexed_vector_store() -> AsyncGenerator[QdrantVectorStore, None]:
    url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_URL")
    assert url, "QDRANT_URL must be set for payload indexes, not supported in memory"

    client = qdrant_client.QdrantClient(url)
    aclient = qdrant_client.AsyncQdrantClient(url)
    collection_name = "test"

    try:
        vector_store = QdrantVectorStore(
            collection_name,
            client=client,
            aclient=aclient,
            payload_indexes=[
                {
                    "field_name": "tenant_id",
                    "field_schema": models.PayloadSchemaType.KEYWORD,
                },
                {
                    "field_name": "some_key",
                    "field_schema": models.PayloadSchemaType.INTEGER,
                },
            ],
            index_doc_id=False,
        )

        nodes = [
            TextNode(
                text="test1",
                id_="11111111-1111-1111-1111-111111111111",
                embedding=[1.0, 0.0],
                metadata={"some_key": 1, "tenant_id": "A"},
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")
                },
            ),
            TextNode(
                text="test2",
                id_="22222222-2222-2222-2222-222222222222",
                embedding=[0.0, 1.0],
                metadata={"some_key": 2, "tenant_id": "B"},
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")
                },
            ),
            TextNode(
                text="test3",
                id_="33333333-3333-3333-3333-333333333333",
                embedding=[1.0, 1.0],
                metadata={"some_key": "3", "tenant_id": "A"},
            ),
        ]

        vector_store.add(nodes)

        # in-memory client does not share data between instances
        await vector_store.async_add(nodes)

        yield vector_store
    finally:
        try:
            await aclient.delete_collection(collection_name=collection_name)
        except Exception:
            pass
        try:
            await aclient.close()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass


@pytest_asyncio.fixture
async def collection_initialized_payload_indexed_vector_store() -> AsyncGenerator[
    QdrantVectorStore, None
]:
    url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_URL")
    assert url, "QDRANT_URL must be set for payload indexes, not supported in memory"

    client = qdrant_client.QdrantClient(url)
    aclient = qdrant_client.AsyncQdrantClient(url)
    collection_name = "test"

    try:
        await aclient.create_collection(
            collection_name,
            vectors_config={
                "text-dense": models.VectorParams(
                    size=2, distance=models.Distance.COSINE
                )
            },
        )

        vector_store = QdrantVectorStore(
            collection_name,
            client=client,
            aclient=aclient,
            payload_indexes=[
                {
                    "field_name": "tenant_id",
                    "field_schema": models.PayloadSchemaType.KEYWORD,
                },
                {
                    "field_name": "some_key",
                    "field_schema": models.PayloadSchemaType.INTEGER,
                },
            ],
            index_doc_id=False,
        )

        nodes = [
            TextNode(
                text="test1",
                id_="11111111-1111-1111-1111-111111111111",
                embedding=[1.0, 0.0],
                metadata={"some_key": 1, "tenant_id": "A"},
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")
                },
            ),
            TextNode(
                text="test2",
                id_="22222222-2222-2222-2222-222222222222",
                embedding=[0.0, 1.0],
                metadata={"some_key": 2, "tenant_id": "B"},
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")
                },
            ),
            TextNode(
                text="test3",
                id_="33333333-3333-3333-3333-333333333333",
                embedding=[1.0, 1.0],
                metadata={"some_key": "3", "tenant_id": "A"},
            ),
        ]

        vector_store.add(nodes)

        # in-memory client does not share data between instances
        await vector_store.async_add(nodes)

        yield vector_store
    finally:
        try:
            await aclient.delete_collection(collection_name=collection_name)
        except Exception:
            pass
        try:
            await aclient.close()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass
