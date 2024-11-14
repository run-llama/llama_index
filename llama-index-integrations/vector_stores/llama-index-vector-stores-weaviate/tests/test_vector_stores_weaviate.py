import weaviate
import pytest

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.weaviate import WeaviateVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in WeaviateVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.fixture(scope="module")
def vector_store():
    client = weaviate.connect_to_embedded()

    vector_store = WeaviateVectorStore(weaviate_client=client)

    nodes = [
        TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
        TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
    ]

    vector_store.add(nodes)

    yield vector_store

    client.close()


def test_basic_flow(vector_store):
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
