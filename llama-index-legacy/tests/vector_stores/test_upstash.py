import os
from importlib.util import find_spec
from typing import List

import pytest
from llama_index.legacy.schema import TextNode
from llama_index.legacy.vector_stores import UpstashVectorStore
from llama_index.legacy.vector_stores.types import VectorStoreQuery

try:
    find_spec("upstash-vector")
    if os.environ.get("UPSTASH_VECTOR_URL") and os.environ.get("UPSTASH_VECTOR_TOKEN"):
        upstash_installed = True
    else:
        upstash_installed = False
except ImportError:
    upstash_installed = False


@pytest.fixture()
def upstash_vector_store() -> UpstashVectorStore:
    return UpstashVectorStore(
        url=os.environ.get("UPSTASH_VECTOR_URL") or "",
        token=os.environ.get("UPSTASH_VECTOR_TOKEN") or "",
    )


@pytest.fixture()
def text_nodes() -> List[TextNode]:
    return [
        TextNode(
            text="llama_index_node_1",
            id_="test_node_1",
            metadata={"hello": "hola"},
            embedding=[0.25] * 256,
        ),
        TextNode(
            text="llama_index_node_2",
            id_="test_node_2",
            metadata={"hello": "hola"},
            embedding=[0.33] * 256,
        ),
    ]


@pytest.mark.skipif(not upstash_installed, reason="upstash-vector not installed")
def test_upstash_vector_add(
    upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode]
) -> None:
    res = upstash_vector_store.add(nodes=text_nodes)
    assert res == ["test_node_1", "test_node_2"]


@pytest.mark.skipif(not upstash_installed, reason="upstash-vector not installed")
def test_upstash_vector_query(
    upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode]
) -> None:
    upstash_vector_store.add(nodes=text_nodes)
    res = upstash_vector_store.query(
        VectorStoreQuery(
            query_embedding=[0.25] * 256,
        )
    )

    assert res.nodes and res.nodes[0].id_ in ["test_node_1", "test_node_2"]
