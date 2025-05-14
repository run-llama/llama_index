from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import pytest_asyncio


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
