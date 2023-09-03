"""Test Epsilla indexes."""
from typing import List

import pytest

try:
    from pyepsilla import vectordb
except ImportError:
    pyepsilla = None  # type: ignore

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import EpsillaVectorStore
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStoreQuery


@pytest.fixture
def node_embeddings() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0, 0.0],
            node=TextNode(
                text="epsilla test text 0.",
                id_="1",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")
                },
                metadata={
                    "date": "2023-08-02",
                },
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 1.0],
            node=TextNode(
                text="epsilla test text 1.",
                id_="2",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")
                },
                metadata={
                    "date": "2023-08-11",
                },
            ),
        ),
    ]


@pytest.mark.skipif(vectordb is None, reason="pyepsilla not installed")
def test_initiate_store() -> None:
    client = vectordb.Client()
    vector_store = EpsillaVectorStore(
        client=client, collection_name="test_collection", dimension=1536
    )

    assert vector_store._collection_created is True
    assert vector_store._collection_name == "test_collection"


@pytest.mark.skipif(vectordb is None, reason="pyepsilla not installed")
def test_add_data_and_query() -> None:
    client = vectordb.Client()
    vector_store = EpsillaVectorStore(client=client, collection_name="test_collection")

    assert vector_store._collection_name == "test_collection"
    assert vector_store._collection_created is not True

    embedding_results = node_embeddings()
    ids = vector_store.add(embedding_results)

    assert vector_store._collection_created is True
    assert ids is ["1", "2"]

    query = VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=1)
    query_result = vector_store.query(query)

    assert query_result.ids is ["1"]
