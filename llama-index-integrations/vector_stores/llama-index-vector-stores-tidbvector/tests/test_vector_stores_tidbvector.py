"""Test TiDB Vector Search functionality."""

from __future__ import annotations

import os
from typing import List

import pytest

try:
    from tidb_vector.integrations import TiDBVectorClient  # noqa

    VECTOR_TABLE_NAME = "llama_index_vector_test"
    CONNECTION_STRING = os.getenv("TEST_TiDB_CONNECTION_URL", "")

    if CONNECTION_STRING == "":
        raise OSError("TEST_TiDB_URL environment variable is not set")

    tidb_available = True
except (OSError, ImportError) as e:
    tidb_available = False


from llama_index.core.schema import (
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

ADA_TOKEN_COUNT = 1536


def text_to_embedding(text: str) -> List[float]:
    """Convert text to a unique embedding using ASCII values."""
    ascii_values = [float(ord(char)) for char in text]
    # Pad or trim the list to make it of length ADA_TOKEN_COUNT
    return ascii_values[:ADA_TOKEN_COUNT] + [0.0] * (
        ADA_TOKEN_COUNT - len(ascii_values)
    )


@pytest.fixture(scope="session")
def node_embeddings() -> list[TextNode]:
    """Return a list of TextNodes with embeddings."""
    return [
        TextNode(
            text="foo",
            id_="f8e7dee2-63b6-42f1-8b60-2d46710c1971",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "theme": "Mystery",
                "location": 11,
            },
            embedding=text_to_embedding("foo"),
        ),
        TextNode(
            text="bar",
            id_="8dde1fbc-2522-4ca2-aedf-5dcb2966d1c6",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={
                "theme": "Friendship",
                "location": 121,
            },
            embedding=text_to_embedding("bar"),
        ),
        TextNode(
            text="baz",
            id_="e4991349-d00b-485c-a481-f61695f2b5ae",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={
                "theme": "Friendship",
                "location": 111,
                "doc_id": "b27abe0a-593b-4b6a-8e3c-1d72889d10eb",
            },
            embedding=text_to_embedding("baz"),
        ),
    ]


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_search(node_embeddings: List[TextNode]) -> None:
    """Test end to end construction and search."""
    tidbvec = TiDBVectorStore(
        table_name=VECTOR_TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    # Add nodes to the tidb vector
    tidbvec.add(node_embeddings)

    # similarity search
    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=1)

    result = tidbvec.query(q)
    tidbvec.drop_vectorstore()
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert result.similarities is not None and result.similarities[0] == 1.0
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_delete_doc(node_embeddings: List[TextNode]) -> None:
    """Test delete document from TiDB Vector Store."""
    tidbvec = TiDBVectorStore(
        table_name=VECTOR_TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    # Add nodes to the tidb vector
    tidbvec.add(node_embeddings)

    # similarity search
    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=3)

    result = tidbvec.query(q)
    assert result.nodes is not None and len(result.nodes) == 3
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert result.similarities is not None and result.similarities[0] == 1.0
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id

    # Delete document
    tidbvec.delete(ref_doc_id="test-0")

    # similarity search
    q = VectorStoreQuery(query_embedding=text_to_embedding("foo"), similarity_top_k=3)

    result = tidbvec.query(q)
    tidbvec.drop_vectorstore()
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[2].text
    )
    assert (
        result.similarities is not None and result.similarities[0] == 0.9953081577931554
    )
    assert result.ids is not None and result.ids[0] == node_embeddings[2].node_id


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_search_with_filter(node_embeddings: List[TextNode]) -> None:
    """Test end to end construction and search with filter."""
    tidbvec = TiDBVectorStore(
        table_name=VECTOR_TABLE_NAME,
        connection_string=CONNECTION_STRING,
        drop_existing_table=True,
    )

    # Add nodes to the tidb vector
    tidbvec.add(node_embeddings)

    # similarity search
    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=1,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=100, operator=">"),
            ]
        ),
    )

    result = tidbvec.query(q)
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[1].text
    )
    assert (
        result.similarities is not None and result.similarities[0] == 0.9977280385800326
    )
    assert result.ids is not None and result.ids[0] == node_embeddings[1].node_id

    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=1,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=100, operator=">"),
                MetadataFilter(key="theme", value="Mystery", operator="=="),
            ],
            condition="and",
        ),
    )

    result = tidbvec.query(q)
    assert result.nodes is not None and len(result.nodes) == 0

    q = VectorStoreQuery(
        query_embedding=text_to_embedding("foo"),
        similarity_top_k=1,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="location", value=100, operator=">"),
                MetadataFilter(key="theme", value="Mystery", operator="=="),
            ],
            condition="or",
        ),
    )

    result = tidbvec.query(q)
    tidbvec.drop_vectorstore()
    assert result.nodes is not None and len(result.nodes) == 1
    assert (
        result.nodes[0].get_content(metadata_mode=MetadataMode.NONE)
        == node_embeddings[0].text
    )
    assert result.similarities is not None and result.similarities[0] == 1.0
    assert result.ids is not None and result.ids[0] == node_embeddings[0].node_id
