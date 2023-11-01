import logging
import os
from typing import Generator

import pytest
from llama_index.schema import TextNode
from llama_index.vector_stores import SingleStoreVectorStore
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
)

logger = logging.getLogger(__name__)

singlestoredb_found = False


@pytest.fixture()
def vector_store() -> Generator[SingleStoreVectorStore, None, None]:
    if "SINGLESTOREDB_URL" in os.environ and "/" in os.environ["SINGLESTOREDB_URL"]:
        url = os.environ["SINGLESTOREDB_URL"]
        table_name = "test"
        singlestoredb_found = True
        store = SingleStoreVectorStore(table_name=table_name)
        store.add(
            [
                TextNode(
                    text="Apples are blue",
                    metadata={"type": "fruit"},
                    embedding=[0.9, 0.1],
                ),
                TextNode(
                    text="Tomatoes are black",
                    metadata={"type": "veggie"},
                    embedding=[0.5, 0.5],
                ),
                TextNode(
                    text="Brownies are orange",
                    metadata={"type": "dessert"},
                    embedding=[0.1, 0.9],
                ),
            ]
        )
        yield store


@pytest.mark.skipif(not singlestoredb_found, reason="singlestoredb not installed")
def test_query(vector_store: SingleStoreVectorStore) -> None:
    result = vector_store.query(
        VectorStoreQuery(query_embedding=[0.9, 0.1], similarity_top_k=1)
    )
    assert result.nodes is not None
    assert len(result.nodes) == 1
    assert isinstance(result.nodes[0], TextNode)
    assert result.nodes[0].text == "Apples are blue"
    assert result.nodes[0].metadata["type"] == "fruit"


@pytest.mark.skipif(not singlestoredb_found, reason="singlestoredb not installed")
def test_metadata_filter(vector_store: SingleStoreVectorStore) -> None:
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
