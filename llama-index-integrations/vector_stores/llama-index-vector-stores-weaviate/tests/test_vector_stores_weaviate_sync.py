import pytest
import weaviate.classes as wvc
import weaviate
from llama_index.core.schema import (
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.vector_stores.weaviate import (
    WeaviateVectorStore,
    AsyncClientNotProvidedError,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


TEST_COLLECTION_NAME = "TestCollection"


def test_class():
    names_of_base_classes = [b.__name__ for b in WeaviateVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.fixture(scope="module")
def client():
    client = weaviate.connect_to_embedded()
    yield client
    client.close()


@pytest.fixture()
def vector_store(client):
    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name=TEST_COLLECTION_NAME
    )
    vector_store.clear()  # Make sure that no leftover test collection exists from a previous test session (embedded Weaviate data gets persisted)
    yield vector_store
    vector_store.clear()


@pytest.fixture()
def vector_store_with_sample_nodes(vector_store):
    nodes = [
        TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
        TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
    ]
    vector_store.add(nodes)
    return vector_store


def test_sync_basic_flow(vector_store_with_sample_nodes):
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="world",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = vector_store_with_sample_nodes.query(query)

    assert len(results.nodes) == 2
    assert results.nodes[0].text == "This is a test."
    assert results.similarities[0] == 1.0

    assert results.similarities[0] > results.similarities[1]


def test_hybrid_search(vector_store_with_sample_nodes):
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.3, 0.0],
        similarity_top_k=10,
        query_str="world",
        mode=VectorStoreQueryMode.HYBRID,
    )

    results = vector_store_with_sample_nodes.query(query)
    assert len(results.nodes) == 2
    assert results.nodes[0].text == "Hello world."
    assert results.nodes[1].text == "This is a test."

    assert results.similarities[0] > results.similarities[1]


def test_query_kwargs(vector_store_with_sample_nodes):
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.3, 0.0],
        similarity_top_k=2,
        query_str="world",
        mode=VectorStoreQueryMode.HYBRID,
    )

    results = vector_store_with_sample_nodes.query(
        query,
        max_vector_distance=0.0,
    )
    assert len(results.nodes) == 0


def test_can_query_collection_with_complex_property_types(client):
    """Verifies that it is possible to query data from collections that contain complex properties (e.g. a list of nested objects in one of the properties)."""
    collection_name = "ComplexTypeInArrayTest"
    client.collections.delete(collection_name)
    collection = client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(
                name="text",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="array_prop",
                data_type=wvc.config.DataType.OBJECT_ARRAY,
                nested_properties=[
                    wvc.config.Property(
                        name="nested_prop",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                ],
            ),
        ],
    )

    collection.data.insert(
        {
            "text": "Text of object containing complex properties",
            "array_prop": [{"nested_prop": "nested_prop content"}],
        },
        vector=[1.0, 0.0, 0.0],
    )

    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=collection_name,
    )
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
        query_str="world",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = vector_store.query(query)

    assert len(results.nodes) == 1
    assert results.nodes[0].text == "Text of object containing complex properties"


def test_sync_delete(vector_store):
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
    vector_store.add(nodes)

    # First check that nothing gets deleted if no matching nodes are present
    vector_store.delete(ref_doc_id="no_match_in_db")
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = vector_store.query(query)
    assert len(results.nodes) == 2

    # Now test actual deletion
    vector_store.delete(ref_doc_id="to_be_deleted")
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = vector_store.query(query)
    assert len(results.nodes) == 1
    results.nodes[0].node_id == node_to_keep.node_id


@pytest.mark.asyncio(loop_scope="module")
async def test_async_methods_called_without_async_client(vector_store):
    """Makes sure that we present an easy to understand error message to the user if he forgets to provide an async client when trying to call async methods."""
    with pytest.raises(AsyncClientNotProvidedError):
        await vector_store.async_add(
            [TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3])]
        )
    with pytest.raises(AsyncClientNotProvidedError):
        await vector_store.adelete(ref_doc_id="no_match_in_db")
    with pytest.raises(AsyncClientNotProvidedError):
        await vector_store.adelete_nodes(node_ids=["sample_node_id"])
    with pytest.raises(AsyncClientNotProvidedError):
        await vector_store.aclear()
    with pytest.raises(AsyncClientNotProvidedError):
        query = VectorStoreQuery(
            query_embedding=[0.3, 0.0, 0.0],
            similarity_top_k=10,
            query_str="test",
            mode=VectorStoreQueryMode.DEFAULT,
        )
        results = await vector_store.aquery(query)
