"""Integration tests for CockroachDBVectorStore against a live CRDB."""

from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore

EMBED_DIM = 4


def _vec(seed: float) -> list[float]:
    return [seed, 1.0, 1.0, 1.0]


def _make_node(node_id: str, text: str, seed: float, **metadata: Any) -> TextNode:
    n = TextNode(id_=node_id, text=text, metadata=metadata)
    n.embedding = _vec(seed)
    n.relationships = {
        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=metadata.get("doc_id", node_id))
    }
    return n


@pytest.fixture()
def store(fresh_db: dict[str, Any]) -> CockroachDBVectorStore:
    s = CockroachDBVectorStore.from_params(
        host=fresh_db["host"],
        port=fresh_db["port"],
        user=fresh_db["user"],
        password=fresh_db["password"] or "",
        database=fresh_db["database"],
        table_name="testidx",
        embed_dim=EMBED_DIM,
        distance_metric="cosine",
        cspann_kwargs={"min_partition_size": 4, "max_partition_size": 16},
        sslmode="disable",
    )
    yield s


def test_add_and_query_default(store: CockroachDBVectorStore) -> None:
    nodes = [
        _make_node("a", "alpha", 1.0, doc_id="d1"),
        _make_node("b", "beta", 0.5, doc_id="d2"),
        _make_node("c", "gamma", 0.0, doc_id="d3"),
    ]
    ids = store.add(nodes)
    assert sorted(ids) == ["a", "b", "c"]

    q = VectorStoreQuery(query_embedding=_vec(1.0), similarity_top_k=2)
    res = store.query(q)
    assert len(res.nodes) == 2
    assert res.ids[0] == "a"


def test_metadata_filter(store: CockroachDBVectorStore) -> None:
    store.add(
        [
            _make_node("a", "alpha", 1.0, category="x"),
            _make_node("b", "beta", 0.9, category="y"),
            _make_node("c", "gamma", 0.8, category="x"),
        ]
    )
    filters = MetadataFilters(
        filters=[MetadataFilter(key="category", value="x", operator=FilterOperator.EQ)]
    )
    res = store.query(
        VectorStoreQuery(query_embedding=_vec(1.0), similarity_top_k=5, filters=filters)
    )
    returned_ids = set(res.ids)
    assert returned_ids == {"a", "c"}


def test_delete_by_ref_doc(store: CockroachDBVectorStore) -> None:
    store.add(
        [
            _make_node("a", "alpha", 1.0, doc_id="doc1"),
            _make_node("b", "beta", 0.9, doc_id="doc1"),
            _make_node("c", "gamma", 0.8, doc_id="doc2"),
        ]
    )
    store.delete("doc1")
    res = store.query(VectorStoreQuery(query_embedding=_vec(1.0), similarity_top_k=10))
    assert res.ids == ["c"]


def test_delete_nodes_by_ids(store: CockroachDBVectorStore) -> None:
    store.add([_make_node(nid, nid, 1.0) for nid in ["a", "b", "c"]])
    store.delete_nodes(node_ids=["a", "b"])
    res = store.query(VectorStoreQuery(query_embedding=_vec(1.0), similarity_top_k=10))
    assert res.ids == ["c"]


def test_clear(store: CockroachDBVectorStore) -> None:
    store.add([_make_node("a", "alpha", 1.0)])
    store.clear()
    res = store.query(VectorStoreQuery(query_embedding=_vec(1.0), similarity_top_k=10))
    assert res.ids == []


def test_get_nodes_returns_embedding(store: CockroachDBVectorStore) -> None:
    store.add([_make_node("a", "alpha", 0.3)])
    nodes = store.get_nodes(node_ids=["a"])
    assert len(nodes) == 1
    assert nodes[0].embedding[0] == pytest.approx(0.3)


def test_mmr_mode_returns_top_k(store: CockroachDBVectorStore) -> None:
    store.add(
        [
            _make_node("a", "alpha", 1.0),
            _make_node("b", "beta", 0.95),
            _make_node("c", "gamma", 0.05),
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_vec(1.0),
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=0.2,
    )
    res = store.query(q, mmr_prefetch_factor=3.0)
    assert len(res.nodes) == 2


def test_unsupported_mode_raises(store: CockroachDBVectorStore) -> None:
    store.add([_make_node("a", "alpha", 1.0)])
    q = VectorStoreQuery(
        query_embedding=_vec(1.0),
        similarity_top_k=1,
        mode=VectorStoreQueryMode.HYBRID,
        query_str="alpha",
    )
    with pytest.raises(NotImplementedError):
        store.query(q)


@pytest.mark.asyncio
async def test_async_add_and_query(store: CockroachDBVectorStore) -> None:
    await store.async_add(
        [
            _make_node("a", "alpha", 1.0),
            _make_node("b", "beta", 0.5),
        ]
    )
    res = await store.aquery(VectorStoreQuery(query_embedding=_vec(1.0), similarity_top_k=1))
    assert res.ids == ["a"]
    await store.close()
