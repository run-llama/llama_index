"""
This tests RocksetVectorStore by creating a new collection,
adding nodes to it, querying nodes, and then
deleting the collection.

To run this test, set ROCKSET_API_KEY and ROCKSET_API_SERVER
env vars. If ROCKSET_API_SERVER is not set, it will use us-west-2.

Find your API server from https://rockset.com/docs/rest-api#introduction.
Get your API key from https://console.rockset.com/apikeys.
"""

from typing import Generator, Any
import pytest

try:
    import rockset

    rockset_installed = True
except ImportError:
    rockset_installed = False
from time import sleep
from llama_index.vector_stores import RocksetVectorStore
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    NodeWithEmbedding,
    VectorStoreQuery,
)
from llama_index.schema import TextNode


def collection_is_empty(client: Any, collection_name: str = "test") -> bool:
    return len(client.sql(f"SELECT _id FROM {collection_name} LIMIT 1").results) == 0


def collection_exists(client: Any, collection_name: str = "test") -> bool:
    try:
        client.Collections.get(collection=collection_name)
    except rockset.exceptions.NotFoundException:
        return False
    return True


@pytest.fixture
def vector_store() -> Generator[RocksetVectorStore, None, None]:
    store = RocksetVectorStore.with_new_collection(collection="test", dimensions=2)
    store = RocksetVectorStore(collection="test")
    store.add(
        [
            NodeWithEmbedding(
                node=TextNode(
                    text="Apples are blue",
                    metadata={"type": "fruit"},  # type: ignore[call-arg]
                ),
                embedding=[0.9, 0.1],
            ),
            NodeWithEmbedding(
                node=TextNode(
                    text="Tomatoes are black",
                    metadata={"type": "veggie"},  # type: ignore[call-arg]
                ),
                embedding=[0.5, 0.5],
            ),
            NodeWithEmbedding(
                node=TextNode(
                    text="Brownies are orange",
                    metadata={"type": "dessert"},  # type: ignore[call-arg]
                ),
                embedding=[0.1, 0.9],
            ),
        ]
    )
    while collection_is_empty(store.client, "test"):  # wait until docs are added
        sleep(0.1)
    yield store
    store.client.Collections.delete(collection="test")
    while collection_exists(store.client, "test"):  # wait until collection is deleted
        sleep(0.1)


@pytest.mark.skipif(not rockset_installed, reason="rockset not installed")
def test_query(vector_store: RocksetVectorStore) -> None:
    result = vector_store.query(
        VectorStoreQuery(query_embedding=[0.9, 0.1], similarity_top_k=1)
    )
    assert result.nodes is not None
    assert len(result.nodes) == 1
    assert isinstance(result.nodes[0], TextNode)
    assert result.nodes[0].text == "Apples are blue"
    assert result.nodes[0].metadata["type"] == "fruit"


@pytest.mark.skipif(not rockset_installed, reason="rockset not installed")
def test_metadata_filter(vector_store: RocksetVectorStore) -> None:
    result = vector_store.query(
        VectorStoreQuery(
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="type", value="dessert")]
            )
        )
    )
    assert result.nodes is not None
    assert len(result.nodes) == 1
    assert isinstance(result.nodes[0], TextNode)
    assert result.nodes[0].text == "Brownies are orange"
    assert result.nodes[0].metadata["type"] == "dessert"
