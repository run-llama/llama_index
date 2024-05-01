import asyncio
import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.vespa import VespaVectorStore, hybrid_template

try:
    # Should be installed as pyvespa-dependency
    import docker

    client = docker.from_env()
    docker_available = client.ping()
except Exception:
    docker_available = False


# Assuming Vespa services are mocked or local Vespa Docker is used
@pytest.fixture(scope="module")
def vespa_app():
    app_package = hybrid_template
    vespa_app = VespaVectorStore(
        application_package=app_package, deployment_target="local"
    )
    yield vespa_app
    vespa_app.client.container.stop()
    vespa_app.client.container.remove()


@pytest.fixture()
def nodes() -> list:
    return [
        TextNode(
            text="The Shawshank Redemption",
            metadata={
                "id": "1",
                "author": "Stephen King",
                "theme": "Friendship",
                "year": 1994,
            },
        ),
        TextNode(
            text="The Godfather",
            metadata={
                "id": "2",
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
                "year": 1972,
            },
        ),
        TextNode(
            text="Inception",
            metadata={
                "id": "3",
                "director": "Christopher Nolan",
                "theme": "Fiction",
                "year": 2010,
            },
        ),
        TextNode(
            text="To Kill a Mockingbird",
            metadata={
                "id": "4",
                "author": "Harper Lee",
                "theme": "Mafia",
                "year": 1960,
            },
        ),
        TextNode(
            text="1984",
            metadata={
                "id": "5",
                "author": "George Orwell",
                "theme": "Totalitarianism",
                "year": 1949,
            },
        ),
        TextNode(
            text="The Great Gatsby",
            metadata={
                "id": "6",
                "author": "F. Scott Fitzgerald",
                "theme": "The American Dream",
                "year": 1925,
            },
        ),
        TextNode(
            text="Harry Potter and the Sorcerer's Stone",
            metadata={
                "id": "7",
                "author": "J.K. Rowling",
                "theme": "Fiction",
                "year": 1997,
            },
        ),
    ]


def test_add_nodes(vespa_app, nodes):
    # Testing the addition of nodes
    ids = vespa_app.add(nodes)
    assert len(ids) == len(nodes)
    assert all(isinstance(id, str) for id in ids)


def test_query_text_search(vespa_app, nodes):
    query = VectorStoreQuery(
        query_str="Inception",  # Ensure the query matches the case used in the nodes
        mode="text_search",
        similarity_top_k=1,
    )
    result = vespa_app.query(query)
    assert len(result.nodes) == 1
    node_metadata = result.nodes[0].metadata
    assert node_metadata["id"] == "3", "Expected Inception node"


def test_query_vector_search(vespa_app, nodes):
    query = VectorStoreQuery(
        query_str="magic, wizardry",
        mode="semantic_hybrid",
        similarity_top_k=1,
    )
    result = vespa_app.query(query)
    assert len(result.nodes) == 1, "Expected 1 result"
    node_metadata = result.nodes[0].metadata
    print(node_metadata)
    assert node_metadata["id"] == "7", "Expected Harry Potter node"


def test_delete_node(vespa_app, nodes):
    # Testing deletion of a node
    vespa_app.delete(ref_doc_id=nodes[0].node_id)
    query = VectorStoreQuery(
        query_str="Shawshank Redemption",
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        similarity_top_k=1,
    )
    result = vespa_app.query(query)
    assert len(result.nodes) == 0


@pytest.mark.asyncio()
async def test_async_add_and_query(vespa_app, nodes):
    # Testing async add and query
    await asyncio.gather(*[vespa_app.async_add(nodes)])
    query = VectorStoreQuery(query_str="Harry Potter", similarity_top_k=1)
    result = await vespa_app.aquery(query)
    assert len(result.nodes) == 1
    assert result.nodes[0].node_id == "7"
