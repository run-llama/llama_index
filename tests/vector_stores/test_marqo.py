import pytest
from llama_index.storage import StorageContext
from llama_index.schema import TextNode
from llama_index.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.marqo import MarqoVectorStore
from llama_index.indices.vector_store.marqo_index import MarqoVectorStoreIndex
from llama_index.readers.marqo import MarqoReader

try:
    import marqo

    marqo.Client().health()
    marqo_not_available = False
except (ImportError, Exception):
    marqo_not_available = True


@pytest.fixture
def marqo_client():
    return marqo.Client(url="http://localhost:8882")


@pytest.fixture
def marqo_vector_store(marqo_client):
    return MarqoVectorStore(index_name="test", marqo_client=marqo_client)


@pytest.mark.skipif(marqo_not_available, reason="marqo is not available")
def test_add_documents(marqo_vector_store):
    # Add some documents to the Marqo vector store
    documents = [
        ("doc1", "This is a test document about cats."),
        ("doc2", "This is a test document about dogs."),
    ]
    ids = marqo_vector_store.add(documents)

    # Check that the returned IDs match the ones we provided
    print(ids)
    assert ids == [doc_id for doc_id, _ in documents]


@pytest.mark.skipif(marqo_not_available, reason="marqo is not available")
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
    print(result)

    # Check that the result contains the ID of the document about cats
    assert "doc1" in result.ids


@pytest.mark.skipif(marqo_not_available, reason="marqo is not available")
def test_add_nodes_to_index(marqo_vector_store):
    # print("test1:", type(marqo_vector_store))  # This line should print "<class 'MarqoVectorStore'>"

    # Create a custom storage context with MarqoVectorStore
    storage_context = StorageContext.from_defaults(vector_store=marqo_vector_store)

    # Create some nodes to add to the index
    nodes = [
        TextNode(text="This is a test document about cats.", id_="doc1"),
        TextNode(text="This is a test document about dogs.", id_="doc2"),
    ]

    marqo_vector_store_index = MarqoVectorStoreIndex(
        storage_context=storage_context, nodes=nodes
    )

    print(
        "test2:", type(marqo_vector_store_index)
    )  # This line should print "<class 'MarqoVectorStoreIndex'>"
    print(
        "test3:", type(marqo_vector_store_index._vector_store)
    )  # This line should print "<class 'MarqoVectorStore'>"

    # Add the nodes to the index
    marqo_vector_store_index._add_nodes_to_index({}, nodes)

    # Query the Marqo vector store to check that the documents were added
    query = VectorStoreQuery(query_str="cats")
    result = marqo_vector_store.query(query)
    print(result)
    assert "doc1" in result.ids


@pytest.mark.skipif(marqo_not_available, reason="marqo is not available")
def test_load_data():
    # Initialize MarqoReader
    marqo_reader = MarqoReader(url="http://localhost:8882")

    # Define the index name and id-to-text map
    index_name = "test"
    id_to_text_map = {
        "doc1": "This is a test document about cats.",
        "doc2": "This is a test document about dogs.",
    }

    # Use MarqoReader to load data with include_vectors= False
    documents = marqo_reader.load_data(
        index_name=index_name,
        id_to_text_map=id_to_text_map,
        top_k=2,
        include_vectors=False,
    )

    # Check that the documents were loaded correctly
    assert len(documents) == 2
    assert {doc.id_ for doc in documents} == set(id_to_text_map.keys())

    # Use MarqoReader to load data with include_vectors= True
    documents = marqo_reader.load_data(
        index_name=index_name,
        id_to_text_map=id_to_text_map,
        top_k=2,
        include_vectors=True,
    )

    # Check that the documents were loaded correctly
    assert len(documents) == 2
    for doc in documents:
        assert doc.embedding is not None
    assert {doc.id_ for doc in documents} == set(id_to_text_map.keys())
