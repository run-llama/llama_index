from llama_index.core.schema import (
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
import pytest
import pytest_asyncio
import weaviate


TEST_COLLECTION_NAME = "TestCollection"


@pytest_asyncio.fixture(scope="module")
async def async_client():
    client = weaviate.use_async_with_embedded()
    await client.connect()
    yield client
    await client.close()


@pytest_asyncio.fixture(loop_scope="module")
async def async_vector_store(async_client):
    vector_store = WeaviateVectorStore(
        weaviate_aclient=async_client, index_name=TEST_COLLECTION_NAME
    )
    await vector_store.aclear()  # Make sure that no leftover test collection exists from a previous test session (embedded Weaviate data gets persisted)
    yield vector_store
    await vector_store.aclear()


@pytest.mark.asyncio(loop_scope="module")
async def test_async_basic_flow(async_vector_store):
    nodes = [
        TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
        TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
    ]

    await async_vector_store.async_add(nodes)

    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = await async_vector_store.aquery(query)

    assert len(results.nodes) == 2
    assert results.nodes[0].text == "This is a test."
    assert results.similarities[0] == 1.0

    assert results.similarities[0] > results.similarities[1]


@pytest.mark.asyncio(loop_scope="module")
async def test_async_old_data_gone(async_vector_store):
    """Makes sure that no data stays in the database in between tests (otherwise more than one node would be found in the assertion)."""
    nodes = [
        TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
    ]

    await async_vector_store.async_add(nodes)

    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=2,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = await async_vector_store.aquery(query)

    assert len(results.nodes) == 1


@pytest.mark.asyncio(loop_scope="module")
async def test_async_delete_nodes(async_vector_store):
    node_to_be_deleted = TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3])
    node_to_keep = TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0])
    nodes = [node_to_be_deleted, node_to_keep]

    await async_vector_store.async_add(nodes)
    await async_vector_store.adelete_nodes(node_ids=[node_to_be_deleted.node_id])
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = await async_vector_store.aquery(query)
    assert len(results.nodes) == 1
    assert results.nodes[0].node_id == node_to_keep.node_id


@pytest.mark.asyncio(loop_scope="module")
async def test_async_delete(async_vector_store):
    node_to_be_deleted = TextNode(
        text="Hello world.",
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_deleted")
        },
        embedding=[0.0, 0.0, 0.3],
    )
    node_to_keep = TextNode(
        text="This is a test.",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="to_be_kept")},
        embedding=[0.3, 0.0, 0.0],
    )
    nodes = [node_to_be_deleted, node_to_keep]
    await async_vector_store.async_add(nodes)

    # First check that nothing gets deleted if no matching nodes are present
    await async_vector_store.adelete(ref_doc_id="no_match_in_db")
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = await async_vector_store.aquery(query)
    assert len(results.nodes) == 2

    # Now test actual deletion
    await async_vector_store.adelete(ref_doc_id="to_be_deleted")
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = await async_vector_store.aquery(query)
    assert len(results.nodes) == 1
    assert results.nodes[0].node_id == node_to_keep.node_id
