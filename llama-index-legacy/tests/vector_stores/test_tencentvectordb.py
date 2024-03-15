import time
from typing import List

import pytest

try:
    import tcvectordb  # noqa: F401

    tcvectordb_init = True
except ImportError:
    tcvectordb_init = False

from llama_index.legacy.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.legacy.vector_stores import TencentVectorDB
from llama_index.legacy.vector_stores.tencentvectordb import (
    CollectionParams,
    FilterField,
)
from llama_index.legacy.vector_stores.types import VectorStoreQuery


@pytest.fixture()
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="test text 1",
            id_="31BA2AA7-E066-452D-B0A6-0935FACE94FC",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-doc-1")
            },
            metadata={"author": "Kiwi", "age": 23},
            embedding=[0.12, 0.32],
        ),
        TextNode(
            text="test text 2",
            id_="38500E76-5436-44A0-9C47-F86AAD56234D",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-doc-2")
            },
            metadata={"author": "Chris", "age": 33},
            embedding=[0.21, 0.22],
        ),
        TextNode(
            text="test text 3",
            id_="9F90A339-2F51-4229-8280-816669102F7F",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-doc-3")
            },
            metadata={"author": "jerry", "age": 41},
            embedding=[0.49, 0.88],
        ),
    ]


def get_tencent_vdb_store(drop_exists: bool = False) -> TencentVectorDB:
    filter_fields = [
        FilterField(name="author"),
        FilterField(name="age", data_type="uint64"),
    ]

    return TencentVectorDB(
        url="http://10.0.X.X",
        key="eC4bLRy2va******************************",
        collection_params=CollectionParams(
            dimension=2, drop_exists=drop_exists, filter_fields=filter_fields
        ),
    )


@pytest.mark.skipif(not tcvectordb_init, reason="`tcvectordb` not installed")
def test_add_stores_data(node_embeddings: List[TextNode]) -> None:
    store = get_tencent_vdb_store(drop_exists=True)
    store.add(node_embeddings)
    time.sleep(2)

    results = store.query_by_ids(
        ["31BA2AA7-E066-452D-B0A6-0935FACE94FC", "38500E76-5436-44A0-9C47-F86AAD56234D"]
    )
    assert len(results) == 2


@pytest.mark.skipif(not tcvectordb_init, reason="`tcvectordb` not installed")
def test_query() -> None:
    store = get_tencent_vdb_store()
    query = VectorStoreQuery(
        query_embedding=[0.21, 0.22],
        similarity_top_k=10,
    )
    result = store.query(query, filter='doc_id in ("test-doc-2", "test-doc-3")')
    assert result.nodes is not None
    assert len(result.nodes) == 2
    assert result.nodes[0].node_id == "38500E76-5436-44A0-9C47-F86AAD56234D"


@pytest.mark.skipif(not tcvectordb_init, reason="`tcvectordb` not installed")
def test_query_with_filter(node_embeddings: List[TextNode]) -> None:
    store = get_tencent_vdb_store()

    query = VectorStoreQuery(
        query_embedding=[0.21, 0.22],
        similarity_top_k=10,
    )

    result = store.query(query, filter="age > 20 and age < 40")
    assert result.nodes is not None
    assert len(result.nodes) == 2
    assert result.nodes[0].metadata.get("author") == "Chris"
    assert result.nodes[1].metadata.get("author") == "Kiwi"


@pytest.mark.skipif(not tcvectordb_init, reason="`tcvectordb` not installed")
def test_delete(node_embeddings: List[TextNode]) -> None:
    ids = [node_embedding.node_id for node_embedding in node_embeddings]

    store = get_tencent_vdb_store()
    results = store.query_by_ids(ids)
    assert len(results) == 3

    store.delete("test-doc-1")
    time.sleep(2)

    results = store.query_by_ids(ids)
    assert len(results) == 2
