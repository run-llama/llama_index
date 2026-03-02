import pytest
from typing import List
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter
from redis import Redis
from redis.asyncio import Redis as RedisAsync

# Using existing Redis container instead of creating a new one
# Make sure Redis is running on localhost:6379


@pytest.fixture()
def dummy_embedding() -> List:
    return [0] * 1536


@pytest.fixture()
def turtle_test() -> dict:
    return {
        "text": "something about turtles",
        "metadata": {"animal": "turtle"},
        "question": "turtle stuff",
        "doc_id": "1234",
    }


@pytest.fixture()
def documents(turtle_test, dummy_embedding) -> List[Document]:
    """
    List of documents represents data to be embedded in the datastore.
    Minimum requirements for Documents in the /upsert endpoint's UpsertRequest.
    """
    return [
        Document(
            text=turtle_test["text"],
            metadata=turtle_test["metadata"],
            doc_id=turtle_test["doc_id"],
            embedding=dummy_embedding,
        ),
        Document(
            text="something about whales",
            metadata={"animal": "whale"},
            doc_id="5678",
            embedding=dummy_embedding,
        ),
    ]


@pytest.fixture()
def test_nodes(documents) -> TextNode:
    parser = SentenceSplitter()
    return parser.get_nodes_from_documents(documents)


@pytest.fixture()
def redis_client() -> Redis:
    return Redis.from_url("redis://localhost:6379/0")


@pytest.fixture()
def redis_client_async() -> RedisAsync:
    """Fixture that provides an asynchronous Redis client."""
    return RedisAsync.from_url("redis://localhost:6379/0")
