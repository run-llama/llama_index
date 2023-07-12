import pytest
from llama_index.schema import TextNode
from llama_index.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.marqo import MarqoVectorStore
from llama_index.indices.vector_store.marqo_index import MarqoVectorStoreIndex
from marqo import Client


@pytest.fixture
def marqo_client():
    return Client(url="http://localhost:8882", api_key="foobar")

@pytest.fixture
def marqo_vector_store(marqo_client):
    return MarqoVectorStore(index_name="test", marqo_client=marqo_client)


def test_add_documents(marqo_vector_store):
    # Add some documents to the Marqo vector store
    documents = [
        ("doc1", "This is a test document about cats."),
        ("doc2", "This is a test document about dogs."),
    ]
    ids = marqo_vector_store.add(documents)

    # Check that the returned IDs match the ones we provided
    assert ids == [doc_id for doc_id, _ in documents]


def test_query(marqo_vector_store):
    # Add some documents to the Marqo vector store
    documents = [
        ("doc1", "This is a test document about cats."),
        ("doc2", "This is a test document about dogs."),
    ]
    marqo_vector_store.add(documents)

    # Query the Marqo vector store
    query = VectorStoreQuery(query_str="cats")
    result = marqo_vector_store.query(query)

    # Check that the result contains the ID of the document about cats
    assert "doc1" in result.ids


def test_add_nodes_to_index(marqo_vector_store):
    # Create a MarqoVectorStoreIndex with the MarqoVectorStore
    marqo_vector_store_index = MarqoVectorStoreIndex(vector_store=marqo_vector_store)

    # Create some nodes to add to the index
    nodes = [
        TextNode(text="This is a test document about cats.", id_="doc1"),
        TextNode(text="This is a test document about dogs.", id_="doc2"),
    ]

    # Add the nodes to the index
    marqo_vector_store_index._add_nodes_to_index({}, nodes)

    # Query the Marqo vector store to check that the documents were added
    query = VectorStoreQuery(query_str="cats")
    result = marqo_vector_store.query(query)
    assert "doc1" in result.ids
