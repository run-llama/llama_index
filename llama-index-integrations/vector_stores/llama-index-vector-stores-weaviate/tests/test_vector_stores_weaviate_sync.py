import pytest
import weaviate.classes as wvc
import weaviate
from llama_index.core.schema import TextNode
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in WeaviateVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.fixture(scope="module")
def client():
    client = weaviate.connect_to_embedded()
    yield client
    client.close()


@pytest.fixture(scope="module")
def vector_store(client):
    vector_store = WeaviateVectorStore(weaviate_client=client)

    nodes = [
        TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
        TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
    ]

    vector_store.add(nodes)

    return vector_store


def test_sync_basic_flow(vector_store):
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=2,
        query_str="world",
        mode=VectorStoreQueryMode.DEFAULT,
    )

    results = vector_store.query(query)

    assert len(results.nodes) == 2
    assert results.nodes[0].text == "This is a test."
    assert results.similarities[0] == 1.0

    assert results.similarities[0] > results.similarities[1]


def test_hybrid_search(vector_store):
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.3, 0.0],
        similarity_top_k=2,
        query_str="world",
        mode=VectorStoreQueryMode.HYBRID,
    )

    results = vector_store.query(query)
    assert len(results.nodes) == 2
    assert results.nodes[0].text == "Hello world."
    assert results.nodes[1].text == "This is a test."

    assert results.similarities[0] > results.similarities[1]


def test_query_kwargs(vector_store):
    query = VectorStoreQuery(
        query_embedding=[0.0, 0.3, 0.0],
        similarity_top_k=2,
        query_str="world",
        mode=VectorStoreQueryMode.HYBRID,
    )

    results = vector_store.query(
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
