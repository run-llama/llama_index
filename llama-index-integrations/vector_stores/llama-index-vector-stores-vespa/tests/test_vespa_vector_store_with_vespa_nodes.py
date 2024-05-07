import pytest
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from vespa.package import ApplicationPackage

from llama_index.schema.vespa.vespa_node import VespaNode
from llama_index.vector_stores.vespa import VespaVectorStore
from llama_index.vector_stores.vespa.templates import with_fields_template


try:
    # Should be installed as pyvespa-dependency
    import docker

    client = docker.from_env()
    docker_available = client.ping()
except Exception:
    docker_available = False

@pytest.fixture(scope="session")
def vespa_app():
    app_package: ApplicationPackage = with_fields_template
    try:
        # Try getting the local instance if available
        return VespaVectorStore(url="http://localhost", application_package=app_package, deployment_target="local")
    except ConnectionError:
        return VespaVectorStore(application_package=app_package, deployment_target="local")
@pytest.fixture(scope="session")
def vespa_nodes() -> list:
    return [
        VespaNode(
            vespa_fields={
                "title": "The Shawshank Redemption",
                "author": "Stephen King",
                "theme": "Friendship",
                "year": 1994,
            },
            text="The Shawshank Redemption",
            metadata={"id": "1"},
        ),
        VespaNode(
            vespa_fields={
                "title": "The Godfather",
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
                "year": 1972,
            },
            text="The Godfather",
            metadata={"id": "2"},
        ),
        VespaNode(
            vespa_fields={
                "title": "Inception",
                "director": "Christopher Nolan",
                "theme": "Fiction",
                "year": 2010,
            },
            text="Inception",
            metadata={"id": "3"},
        ),
        VespaNode(
            vespa_fields={
                "title": "To Kill a Mockingbird",
                "author": "Harper Lee",
                "theme": "Mafia",
                "year": 1960,
            },
            text="To Kill a Mockingbird",
            metadata={"id": "4"},
        ),
        VespaNode(
            vespa_fields={
                "title": "1984",
                "author": "George Orwell",
                "theme": "Totalitarianism",
                "year": 1949,
            },
            text="1984",
            metadata={"id": "5"},
        ),
        VespaNode(
            vespa_fields={
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "theme": "The American Dream",
                "year": 1925,
            },
            text="The Great Gatsby",
            metadata={"id": "6"},
        ),
        VespaNode(
            vespa_fields={
                "title": "Harry Potter and the Sorcerer's Stone",
                "author": "J.K. Rowling",
                "theme": "Fiction",
                "year": 1997,
            },
            text="Harry Potter and the Sorcerer's Stone",
            metadata={"id": "7"},
        ),

    ]

@pytest.fixture(scope="session")
def added_node_ids(vespa_app, vespa_nodes):
    return vespa_app.add(vespa_nodes)

@pytest.mark.skipif(not docker_available, reason="Docker not available")
def test_vespa_node_text_query(vespa_app, added_node_ids):
    query = VectorStoreQuery(
        query_str="Gatsby",  # Ensure the query matches the case used in the nodes
        mode=VectorStoreQueryMode.DEFAULT,
        similarity_top_k=1,
    )
    result = vespa_app.query(query)
    assert len(result.nodes) == 1
    print(result.nodes[0].metadata)
