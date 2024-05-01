import asyncio
import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.vespa import VespaVectorStore, hybrid_template


# Assuming Vespa services are mocked or local Vespa Docker is used
@pytest.fixture(scope="module")
def vespa_app():
    app_package = hybrid_template
    return VespaVectorStore(application_package=app_package, deployment_target="local")


@pytest.fixture()
def nodes() -> list:
    return [
        TextNode(
            text="The Shawshank Redemption",
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
                "year": 1994,
            },
        ),
        TextNode(
            text="The Godfather",
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
                "year": 1972,
            },
        ),
        TextNode(
            text="Inception",
            metadata={
                "director": "Christopher Nolan",
                "theme": "Fiction",
                "year": 2010,
            },
        ),
        TextNode(
            text="To Kill a Mockingbird",
            metadata={
                "author": "Harper Lee",
                "theme": "Mafia",
                "year": 1960,
            },
        ),
        TextNode(
            text="1984",
            metadata={
                "author": "George Orwell",
                "theme": "Totalitarianism",
                "year": 1949,
            },
        ),
        TextNode(
            text="The Great Gatsby",
            metadata={
                "author": "F. Scott Fitzgerald",
                "theme": "The American Dream",
                "year": 1925,
            },
        ),
        TextNode(
            text="Harry Potter and the Sorcerer's Stone",
            metadata={
                "author": "J.K. Rowling",
                "theme": "Fiction",
                "year": 1997,
            },
        ),
    ]


def test_add_nodes(vespa_app, nodes):
    # Testing the addition of nodes
    ids = vespa_app.add(nodes)
    assert len(ids) == 2
    assert all(isinstance(id, str) for id in ids)


def test_query(vespa_app):
    # Testing querying
    query = VectorStoreQuery(
        query_str="test", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=1
    )
    result = vespa_app.query(query)
    assert len(result.nodes) > 0
    assert result.nodes[0].get_content().contains("test")


def test_delete_node(vespa_app, nodes):
    # Testing deletion of a node
    vespa_app.add(nodes)
    vespa_app.delete(ref_doc_id=nodes[0].node_id)
    query = VectorStoreQuery(
        query_str="This is a test.",
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        similarity_top_k=1,
    )
    result = vespa_app.query(query)
    assert not any(node.node_id == nodes[0].node_id for node in result.nodes)


@pytest.mark.asyncio()
async def test_async_add_and_query(vespa_app, nodes):
    # Testing async add and query
    await asyncio.gather(*[vespa_app.async_add(nodes)])
    query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=1)
    result = await vespa_app.aquery(query)
    assert len(result.nodes) == 1
    assert result.nodes[0].node_id == "1"
