import pytest
from llama_index.storage import StorageContext
from llama_index.schema import TextNode
from llama_index.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.marqo import MarqoVectorStore
from llama_index.indices.vector_store.marqo_index import MarqoVectorStoreIndex
from llama_index.vector_stores.types import NodeWithEmbedding
from llama_index.data_structs.data_structs import IndexDict
from llama_index.readers.marqo import MarqoReader
from marqo import Client

try:
    import marqo

    marqo.Client().health()
    marqo_not_available = False
except (ImportError, Exception):
    marqo_not_available = True


@pytest.fixture
def marqo_client() -> Client:
    return Client(url="http://localhost:8882")


@pytest.fixture
def marqo_vector_store(marqo_client: Client) -> MarqoVectorStore:
    return MarqoVectorStore(index_name="test", marqo_client=marqo_client)


@pytest.mark.skipif(marqo_not_available, reason="marqo is not available")
def test_add_documents(
    marqo_vector_store: MarqoVectorStore, marqo_client: Client
) -> None:
    # Add some documents to the Marqo vector store
    documents = [
        NodeWithEmbedding(
            node=TextNode(id_="doc1", text="This is a test document about cats."),
            embedding=[],
        ),
        NodeWithEmbedding(
            node=TextNode(id_="doc2", text="This is a test document about dogs."),
            embedding=[],
        ),
    ]
    ids = marqo_vector_store.add(documents=documents)

    # Check that the returned IDs match the ones we provided
    assert ids == [doc.node.node_id for doc in documents]


@pytest.mark.skipif(marqo_not_available, reason="marqo is not available")
def test_query(marqo_vector_store: MarqoVectorStore) -> None:
    # Add some documents to the Marqo vector store
    documents = [
        NodeWithEmbedding(
            node=TextNode(id_="doc1", text="This is a test document about cats."),
            embedding=[],
        ),
        NodeWithEmbedding(
            node=TextNode(id_="doc2", text="This is a test document about dogs."),
            embedding=[],
        ),
    ]
    marqo_vector_store.add(documents=documents)

    # Query the Marqo vector store
    query = VectorStoreQuery(query_str="cats")
    result = marqo_vector_store.query(query)

    # Check that the result contains the ID of the document about cats
    assert "doc1" in (result.ids or [])


@pytest.mark.skipif(marqo_not_available, reason="marqo is not available")
def test_add_nodes_to_index(marqo_vector_store: MarqoVectorStore) -> None:
    # Create a custom storage context with MarqoVectorStore
    storage_context = StorageContext.from_defaults(vector_store=marqo_vector_store)

    # Create some nodes to add to the index
    nodes = [
        TextNode(text="This is a test document about cats.", id_="doc1", embedding=[]),
        TextNode(text="This is a test document about dogs.", id_="doc2", embedding=[]),
    ]

    marqo_vector_store_index = MarqoVectorStoreIndex(
        storage_context=storage_context, nodes=nodes
    )

    # Add the nodes to the index
    marqo_vector_store_index._add_nodes_to_index(IndexDict(), nodes)

    # Query the Marqo vector store to check that the documents were added
    query = VectorStoreQuery(query_str="cats")
    result = marqo_vector_store.query(query)
    assert "doc1" in (result.ids or [])


@pytest.mark.skipif(marqo_not_available, reason="marqo is not available")
def test_load_data() -> None:
    # Initialize MarqoReader
    marqo_reader = MarqoReader(url="http://localhost:8882")

    # Define the index name and searchable_attributes
    index_name = "test"

    # Use MarqoReader to load data with include_vectors= False
    documents = marqo_reader.load_data(
        index_name=index_name,
        top_k=2,
        include_vectors=False,
    )

    # Check that the documents were loaded correctly
    assert len(documents) == 2

    # Use MarqoReader to load data with include_vectors= True
    documents = marqo_reader.load_data(
        index_name=index_name,
        top_k=2,
        include_vectors=True,
    )

    # Check that the documents were loaded correctly
    assert len(documents) == 2
    for doc in documents:
        assert doc.embedding is not None
