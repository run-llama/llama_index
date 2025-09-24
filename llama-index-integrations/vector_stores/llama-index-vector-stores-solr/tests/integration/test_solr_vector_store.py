"""
end-to-end integration tests for ApacheSolrVectorStore.

Functionality Covered:
1. Index (add / async_add) a handful of nodes with metadata + embeddings.
2. Retrieve via:
   - Match all (*:*) raw query (implicit through vector store query without embedding)
   - Dense KNN query (VectorStoreQueryMode.DEFAULT)
   - Lexical BM25 query (VectorStoreQueryMode.TEXT_SEARCH)
   - Same dense & lexical queries but with metadata filters applied.
3. Delete one node and validate it is gone using *:*
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc
from typing import Callable

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.solr.base import ApacheSolrVectorStore
from llama_index.vector_stores.solr.client.async_ import AsyncSolrClient
from llama_index.vector_stores.solr.client.sync import SyncSolrClient
from llama_index.vector_stores.solr.types import BoostedTextField

# ---------------------------------------------------------------------------
# Fixtures (local to this file to keep file self-contained)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_nodes() -> list[TextNode]:
    """
    Return nodes with simple hardcoded embeddings (64-dim) + metadata.

    Embedding design:
      n1: all 0.125 -> close to history query [0.15]*64
      n2: all 0.25 -> also close to history query
      n3: all 0.90 -> close to politics query [0.85]*64
    This yields deterministic nearest-neighbor expectations.
    """
    base_dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return [
        TextNode(
            id_="n1",
            text="Abraham Lincoln was President of the United States.",
            embedding=[0.125] * 64,
            metadata={"topic": "history", "published": base_dt},
        ),
        TextNode(
            id_="n2",
            text="Benjamin Franklin was a Founding Father, not a President.",
            embedding=[0.25] * 64,
            metadata={"topic": "history", "published": base_dt.replace(day=2)},
        ),
        TextNode(
            id_="n3",
            text="John Major served as Prime Minister of the United Kingdom.",
            embedding=[0.9] * 64,
            metadata={"topic": "politics", "published": base_dt.replace(day=3)},
        ),
    ]


@pytest.fixture()
def vector_store(
    function_unique_solr_with_knn_collection_url: str,
) -> ApacheSolrVectorStore:
    """
    Create a real vector store hitting the per-test Solr collection.

    We configure both dense (embedding) and lexical (BM25) search fields.
    """
    sync_client = SyncSolrClient(base_url=function_unique_solr_with_knn_collection_url)
    async_client = AsyncSolrClient(
        base_url=function_unique_solr_with_knn_collection_url
    )

    return ApacheSolrVectorStore(
        sync_client=sync_client,
        async_client=async_client,
        nodeid_field="id",
        docid_field="docid",
        content_field="text_txt_en",
        embedding_field="vector_field",  # dense vector field configured in test Solr schema see conftest.py
        metadata_to_solr_field_mapping=[
            ("topic", "topic_s"),
            ("published", "published_dt"),
        ],
        text_search_fields=[BoostedTextField(field="text_txt_en", boost_factor=2.0)],
        solr_field_preprocessor_kwargs={},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poll_count(
    store: ApacheSolrVectorStore,
    target_pred: Callable[[int], bool],
    *,
    max_attempts: int = 30,
    sleep_s: float = 0.2,
) -> int:
    """
    Poll Solr until predicate satisfied or timeout.

    Returns final count or raises AssertionError.
    """
    last_count = -1
    for _ in range(max_attempts):
        resp = store.sync_client.search({"q": "*:*"})
        last_count = len(resp.response.docs)
        if target_pred(last_count):
            return last_count
        time.sleep(sleep_s)
    raise AssertionError(f"Timeout waiting for condition; last_count={last_count}")


def _build_dense_query(embedding: list[float], top_k: int = 3) -> VectorStoreQuery:
    return VectorStoreQuery(
        mode=VectorStoreQueryMode.DEFAULT,
        query_embedding=embedding,
        similarity_top_k=top_k,
    )


def _build_text_query(q: str, top_k: int = 3) -> VectorStoreQuery:
    return VectorStoreQuery(
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        query_str=q,
        sparse_top_k=top_k,
    )


def _topic_filter(value: str) -> MetadataFilters:
    return MetadataFilters(
        filters=[MetadataFilter(key="topic_s", value=value, operator=FilterOperator.EQ)]
    )


# Predefined query embeddings for deterministic dense search
DEFAULT_EMBEDDINGS = {
    "history_query": [0.15] * 64,  # closer to n1/n2
    "politics_query": [0.85] * 64,  # closer to n3
    "founding_father_query": [0.18] * 64,  # still closer to n1/n2
}


# ---------------------------------------------------------------------------
# Synchronous end-to-end test
# ---------------------------------------------------------------------------


@pytest.mark.uses_docker
def test_solr_vector_store_sync_minimal_flow(
    vector_store: ApacheSolrVectorStore,
    sample_nodes: list[TextNode],
) -> None:
    """
    End-to-end sync flow covering add, * query, dense & lexical, filters, delete.

    Steps:
      1. add() nodes
      2. verify they are visible via raw *:*
      3. dense KNN query
      4. lexical BM25 query
      5. filtered dense + lexical queries
      6. delete one node (using delete_nodes since we mapped node ids, simpler)
      7. verify count decreased
    """
    # 1. Index
    added_ids = vector_store.add(sample_nodes)
    assert added_ids == [n.id_ for n in sample_nodes]

    # 2. Ensure visibility
    _poll_count(vector_store, lambda c: c == len(sample_nodes))

    # 3. Dense KNN query (predefined query embedding)
    dense_q = _build_dense_query(DEFAULT_EMBEDDINGS["history_query"], top_k=3)
    dense_res = vector_store.query(dense_q)
    assert len(dense_res.ids) <= 3
    # expect at least one history doc (n1/n2) retrieved
    assert any(doc_id in dense_res.ids for doc_id in ["n1", "n2"])

    # 4. Lexical BM25 query
    text_q = _build_text_query("President", top_k=3)
    text_res = vector_store.query(text_q)
    assert len(text_res.ids) >= 1

    # 5. Filtered queries
    hist_filter = _topic_filter("history")

    dense_hist_q = _build_dense_query(DEFAULT_EMBEDDINGS["history_query"], top_k=5)
    dense_hist_q.filters = hist_filter
    dense_hist_res = vector_store.query(dense_hist_q)
    assert set(dense_hist_res.ids).issubset({"n1", "n2"})

    text_hist_q = _build_text_query("President", top_k=5)
    text_hist_q.filters = hist_filter
    text_hist_res = vector_store.query(text_hist_q)
    assert set(text_hist_res.ids).issubset({"n1", "n2"})

    # 6. Delete one node by node id
    vector_store.delete_nodes(node_ids=["n1"])
    _poll_count(vector_store, lambda c: c == len(sample_nodes) - 1)

    # 7. Final assertion: n1 removed
    remaining = vector_store.sync_client.search({"q": "*:*"}).response.docs
    remaining_ids = {d["id"] for d in remaining}
    assert "n1" not in remaining_ids


# ---------------------------------------------------------------------------
# Asynchronous end-to-end test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.uses_docker
async def test_solr_vector_store_async_minimal_flow(
    vector_store: ApacheSolrVectorStore,
    sample_nodes: list[TextNode],
) -> None:
    """
    Async variant of the minimal flow using async_add, aquery, adelete_nodes.
    """
    # 1. Index (async)
    added_ids = await vector_store.async_add(sample_nodes)
    assert added_ids == [n.id_ for n in sample_nodes]

    _poll_count(vector_store, lambda c: c == len(sample_nodes))

    # Dense query via encoded text
    dense_q = _build_dense_query(DEFAULT_EMBEDDINGS["politics_query"], top_k=3)
    dense_res = await vector_store.aquery(dense_q)
    # Expect politics doc n3 likely present
    assert "n3" in dense_res.ids

    # Lexical query (BM25)
    text_q = _build_text_query("Minister", top_k=3)
    text_res = await vector_store.aquery(text_q)
    assert len(text_res.ids) >= 1

    # Filtered dense query for history topic
    hist_filter = _topic_filter("history")
    dense_hist_q = _build_dense_query(
        DEFAULT_EMBEDDINGS["founding_father_query"], top_k=5
    )
    dense_hist_q.filters = hist_filter
    dense_hist_res = await vector_store.aquery(dense_hist_q)
    assert set(dense_hist_res.ids).issubset({"n1", "n2"})

    # Delete one node
    await vector_store.adelete_nodes(node_ids=["n2"])  # remove a history doc
    _poll_count(vector_store, lambda c: c == len(sample_nodes) - 1)

    remaining = vector_store.sync_client.search({"q": "*:*"}).response.docs
    remaining_ids = {d["id"] for d in remaining}
    assert "n2" not in remaining_ids
