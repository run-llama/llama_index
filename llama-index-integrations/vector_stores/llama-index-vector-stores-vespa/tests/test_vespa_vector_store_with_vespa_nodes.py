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
                "id": "1"
            },
            text="The Shawshank Redemption",
            metadata={"added_by": "gokturkDev"},
        ),
        VespaNode(
            vespa_fields={
                "title": "The Godfather",
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
                "year": 1972,
                "id": "2"
            },
            text="The Godfather",
            metadata={"added_by": "gokturkDev"},
        ),
        VespaNode(
            vespa_fields={
                "title": "Inception",
                "director": "Christopher Nolan",
                "theme": "Fiction",
                "year": 2010,
                "id": "3"
            },
            text="Inception",
            metadata={"added_by": "gokturkDev"},
        ),
        VespaNode(
            vespa_fields={
                "title": "To Kill a Mockingbird",
                "author": "Harper Lee",
                "theme": "Mafia",
                "year": 1960,
                "id": 4
            },
            text="To Kill a Mockingbird",
            metadata={"added_by": "gokturkDev"},
        ),
        VespaNode(
            vespa_fields={
                "title": "1984",
                "author": "George Orwell",
                "theme": "Totalitarianism",
                "year": 1949,
                "id": "5"
            },
            text="1984",
            metadata={"added_by": "gokturkDev"},
        ),
        VespaNode(
            vespa_fields={
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "theme": "The American Dream",
                "year": 1925,
                "id": "6"
            },
            text="The Great Gatsby",
            metadata={"added_by": "gokturkDev"},
        ),
        VespaNode(
            vespa_fields={
                "title": "Harry Potter and the Sorcerer's Stone",
                "author": "J.K. Rowling",
                "theme": "Fiction",
                "year": 1997,
                "id": "7"
            },
            text="Harry Potter and the Sorcerer's Stone",
            metadata={"added_by": "gokturkDev"},
        ),

    ]

@pytest.fixture(scope="session")
def added_node_ids(vespa_app, vespa_nodes):
    yield vespa_app.add(vespa_nodes)
    for node in vespa_nodes:
        vespa_app.delete(node.node_id)

class TestTextQuery:
    def setup_method(self):
        self.text_query =  VectorStoreQuery(
            query_str="1984",  # Ensure the query matches the case used in the nodes
            mode=VectorStoreQueryMode.DEFAULT,
            similarity_top_k=1,
        )
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_returns_hit(self, vespa_app, added_node_ids):
        result = vespa_app.query(self.text_query)
        assert len(result.nodes) == 1

    def test_returns_vespa_node(self, vespa_app, added_node_ids):
        result = vespa_app.query(self.text_query)
        node = result.nodes[0]
        assert isinstance(node, VespaNode)
    def test_correctly_assigns_vespa_node_fields(self, vespa_app, added_node_ids):
        result = vespa_app.query(self.text_query)
        node = result.nodes[0]
        assert node.vespa_fields["title"] == "1984"
        assert node.vespa_fields["author"] == "George Orwell"
        assert node.vespa_fields["theme"] == "Totalitarianism"
        assert node.vespa_fields["year"] == 1949
        assert node.vespa_fields["id"] == "5"