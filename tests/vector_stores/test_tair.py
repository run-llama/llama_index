from typing import List
from os import environ

import pytest

try:
    from tair import Tair
except ImportError:
    Tair = None  # type: ignore

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores import TairVectorStore
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStoreQuery


@pytest.fixture
def node_embeddings() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0, 0.0],
            node=Node(
                text="lorem ipsum",
                doc_id="AF3BE6C4-5F43-4D74-B075-6B0E07900DE8",
                relationships={DocumentRelationship.SOURCE: "test-0"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0, 1.0],
            node=Node(
                text="lorem ipsum",
                doc_id="7D9CD555-846C-445C-A9DD-F8924A01411D",
                relationships={DocumentRelationship.SOURCE: "test-1"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[1.0, 1.0],
            node=Node(
                text="lorem ipsum",
                doc_id="452D24AB-F185-414C-A352-590B4B9EE51B",
                relationships={DocumentRelationship.SOURCE: "test-1"},
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
    assert len(result.ids) == 1
    assert result.ids[0] == "452D24AB-F185-414C-A352-590B4B9EE51B"


@pytest.mark.skipif(Tair is None, reason="tair-py not installed")
def test_delete() -> None:
    tair_url = get_tair_url()
    tair_vector_store = TairVectorStore(tair_url=tair_url, index_name="test_index")

    tair_vector_store.delete("test-1")
    info = tair_vector_store.client.tvs_get_index("test_index")
    assert int(info["data_count"]) == 1

    query = VectorStoreQuery(query_embedding=[1.0, 1.0])
    result = tair_vector_store.query(query)
    assert len(result.ids) == 1
    assert result.ids[0] == "AF3BE6C4-5F43-4D74-B075-6B0E07900DE8"

    tair_vector_store.delete_index()
    info = tair_vector_store.client.tvs_get_index("test_index")
    assert info is None
