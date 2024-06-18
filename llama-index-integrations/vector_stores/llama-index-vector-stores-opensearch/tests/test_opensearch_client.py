import asyncio
import logging
import pytest
import uuid
from typing import List, Generator

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from llama_index.core.vector_stores.types import VectorStoreQuery

##
# Start Opensearch locally
# cd tests
# docker-compose up
#
# Run tests
# pytest test_opensearch_client.py

logging.basicConfig(level=logging.DEBUG)
evt_loop = asyncio.get_event_loop()

try:
    from opensearchpy import AsyncOpenSearch

    os_client = AsyncOpenSearch("localhost:9200")
    evt_loop.run_until_complete(os_client.info())
    opensearch_not_available = False
except (ImportError, Exception):
    opensearch_not_available = True
finally:
    evt_loop.run_until_complete(os_client.close())


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_connection() -> None:
    assert True


@pytest.fixture()
def index_name() -> str:
    """Return the index name."""
    return f"test_{uuid.uuid4().hex}"


@pytest.fixture()
def os_store(index_name: str) -> Generator[OpensearchVectorStore, None, None]:
    client = OpensearchVectorClient(
        endpoint="localhost:9200",
        index=index_name,
        dim=3,
    )

    yield OpensearchVectorStore(client)

    # teardown step
    # delete index
    evt_loop.run_until_complete(client._os_client.indices.delete(index=index_name))
    # close client aiohttp session
    evt_loop.run_until_complete(client._os_client.close())


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "author": "Stephen King",
                "theme": "Friendship",
            },
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="lorem ipsum",
            id_="c3ew11cd-8fb4-4b8f-b7ea-7fa96038d39d",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "director": "Christopher Nolan",
            },
            embedding=[0.0, 0.0, 1.0],
        ),
        TextNode(
            text="I was taught that the way of progress was neither swift nor easy.",
            id_="0b31ae71-b797-4e88-8495-031371a7752e",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-3")},
            metadata={
                "author": "Marie Curie",
            },
            embedding=[0.0, 0.0, 0.9],
        ),
        TextNode(
            text=(
                "The important thing is not to stop questioning."
                + " Curiosity has its own reason for existing."
            ),
            id_="bd2e080b-159a-4030-acc3-d98afd2ba49b",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-4")},
            metadata={
                "author": "Albert Einstein",
            },
            embedding=[0.0, 0.0, 0.5],
        ),
        TextNode(
            text=(
                "I am no bird; and no net ensnares me;"
                + " I am a free human being with an independent will."
            ),
            id_="f658de3b-8cef-4d1c-8bed-9a263c907251",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-5")},
            metadata={
                "author": "Charlotte Bronte",
            },
            embedding=[0.0, 0.0, 0.3],
        ),
    ]


def count_docs_in_index(os_store: OpensearchVectorStore) -> int:
    """Refresh indices and return the count of documents in the index."""
    evt_loop.run_until_complete(
        os_store.client._os_client.indices.refresh(index=os_store.client._index)
    )
    count = evt_loop.run_until_complete(
        os_store.client._os_client.count(index=os_store.client._index)
    )
    return count["count"]


@pytest.mark.skipif(opensearch_not_available, reason="opensearch is not available")
def test_functionality(
    os_store: OpensearchVectorStore, node_embeddings: List[TextNode]
) -> None:
    # add
    assert len(os_store.add(node_embeddings)) == len(node_embeddings)
    # query
    exp_node = node_embeddings[3]
    query = VectorStoreQuery(query_embedding=exp_node.embedding, similarity_top_k=1)
    query_result = os_store.query(query)
    assert query_result.nodes
    assert query_result.nodes[0].get_content() == exp_node.text
    # delete one node using its associated doc_id
    os_store.delete("test-1")
    assert count_docs_in_index(os_store) == len(node_embeddings) - 1
