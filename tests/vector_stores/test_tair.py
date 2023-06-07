from os import environ
from typing import List

import pytest

try:
    from tair import Tair
except ImportError:
    Tair = None  # type: ignore

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores import TairVectorStore
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    NodeWithEmbedding,
    VectorStoreQuery,
)


@pytest.fixture
def node_embeddings() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0, 0.0],
            node=Node(
                text="lorem ipsum",
                doc_id="AF3BE6C4-5F43-4D74-B075-6B0E07900DE8",
                relationships={DocumentRelationship.SOURCE: "test-0"},
                extra_info={"weight": 1.0, "rank": "a"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 1.0],
            node=Node(
                text="lorem ipsum",
                doc_id="7D9CD555-846C-445C-A9DD-F8924A01411D",
                relationships={DocumentRelationship.SOURCE: "test-1"},
                extra_info={"weight": 2.0, "rank": "c"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[1.0, 1.0],
            node=Node(
                text="lorem ipsum",
                doc_id="452D24AB-F185-414C-A352-590B4B9EE51B",
                relationships={DocumentRelationship.SOURCE: "test-1"},
                extra_info={"weight": 3.0, "rank": "b"},
            ),
        ),
    ]


def get_tair_url() -> str:
    return environ.get("TAIR_URL", "redis://localhost:6379")


@pytest.mark.skipif(Tair is None, reason="tair-py not installed")
def test_add_stores_data(node_embeddings: List[NodeWithEmbedding]) -> None:
    tair_url = get_tair_url()
    tair_vector_store = TairVectorStore(tair_url=tair_url, index_name="test_index")

    tair_vector_store.add(node_embeddings)

    info = tair_vector_store.client.tvs_get_index("test_index")
    assert int(info["data_count"]) == 3


@pytest.mark.skipif(Tair is None, reason="tair-py not installed")
def test_query() -> None:
    tair_url = get_tair_url()
    tair_vector_store = TairVectorStore(tair_url=tair_url, index_name="test_index")

    query = VectorStoreQuery(query_embedding=[1.0, 1.0])
    result = tair_vector_store.query(query)
    assert (
        result.ids is not None
        and len(result.ids) == 1
        and result.ids[0] == "452D24AB-F185-414C-A352-590B4B9EE51B"
    )

    # query with filters
    filters = MetadataFilters(filters=[ExactMatchFilter(key="rank", value="c")])
    query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
    result = tair_vector_store.query(query)
    assert (
        result.ids is not None
        and len(result.ids) == 1
        and result.ids[0] == "7D9CD555-846C-445C-A9DD-F8924A01411D"
    )

    filters = MetadataFilters(filters=[ExactMatchFilter(key="weight", value=1.0)])
    filters.filters[0].value = 1.0
    query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
    result = tair_vector_store.query(query)
    assert (
        result.ids is not None
        and len(result.ids) == 1
        and result.ids[0] == "AF3BE6C4-5F43-4D74-B075-6B0E07900DE8"
    )

    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="rank", value="c"),
            ExactMatchFilter(key="weight", value=1.0),
        ]
    )
    query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
    result = tair_vector_store.query(query)
    assert result.ids is not None and len(result.ids) == 0

    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="rank", value="a"),
            ExactMatchFilter(key="weight", value=1.0),
        ]
    )
    query = VectorStoreQuery(query_embedding=[1.0, 1.0], filters=filters)
    result = tair_vector_store.query(query)
    assert (
        result.ids is not None
        and len(result.ids) == 1
        and result.ids[0] == "AF3BE6C4-5F43-4D74-B075-6B0E07900DE8"
    )


@pytest.mark.skipif(Tair is None, reason="tair-py not installed")
def test_delete() -> None:
    tair_url = get_tair_url()
    tair_vector_store = TairVectorStore(tair_url=tair_url, index_name="test_index")

    tair_vector_store.delete("test-1")
    info = tair_vector_store.client.tvs_get_index("test_index")
    assert int(info["data_count"]) == 1

    query = VectorStoreQuery(query_embedding=[1.0, 1.0])
    result = tair_vector_store.query(query)
    assert (
        result.ids is not None
        and len(result.ids) == 1
        and result.ids[0] == "AF3BE6C4-5F43-4D74-B075-6B0E07900DE8"
    )

    tair_vector_store.delete_index()
    info = tair_vector_store.client.tvs_get_index("test_index")
    assert info is None
