"""
Unit tests for MilvusVectorStore.

Tests cover add, query, delete, and async variants using mocked Milvus clients.
The async sparse search fix (async_encode_queries) is specifically covered.
"""

import uuid
from contextlib import contextmanager
from typing import Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction

DIM = 4
COLLECTION = "test_milvus_collection"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_node(text: str = "hello world", dim: int = DIM) -> TextNode:
    return TextNode(
        node_id=str(uuid.uuid4()),
        text=text,
        embedding=[0.1] * dim,
        metadata={"category": "test"},
    )


@contextmanager
def _patched_milvus(
    mock_sync: MagicMock, mock_async: AsyncMock
) -> Generator[None, None, None]:
    """Patch both Milvus clients for the duration of a `with` block."""
    with (
        patch(
            "llama_index.vector_stores.milvus.base.MilvusClient",
            return_value=mock_sync,
        ),
        patch(
            "llama_index.vector_stores.milvus.base.AsyncMilvusClient",
            return_value=mock_async,
        ),
    ):
        yield


def _make_milvus_hit(node: TextNode, distance: float = 0.9) -> dict:
    """
    Build a fake Milvus search hit matching what _parse_from_milvus_results expects.

    The Milvus SDK returns hit objects that behave like dicts:
      hit["id"], hit["distance"], hit["entity"][field], "field" in hit["entity"]
    """
    return {
        "id": node.node_id,
        "distance": distance,
        "entity": {
            "id": node.node_id,
            "text": node.text,
            "doc_id": node.node_id,
            "_node_type": "TextNode",
            "_node_content": node.json(),
        },
    }


class FakeSparseEmbeddingFunction(BaseSparseEmbeddingFunction):
    """Minimal sparse embedding function for testing."""

    def encode_queries(self, queries: List[str]) -> List[Dict[int, float]]:
        return [{0: 0.5, 1: 0.3} for _ in queries]

    def encode_documents(self, documents: List[str]) -> List[Dict[int, float]]:
        return [{0: 0.5, 1: 0.3} for _ in documents]

    async def async_encode_queries(self, queries: List[str]) -> List[Dict[int, float]]:
        return self.encode_queries(queries)


@pytest.fixture()
def mock_milvus_client():
    """Return a MagicMock that stands in for MilvusClient."""
    client = MagicMock()
    client.list_collections.return_value = []
    client.create_schema.return_value = MagicMock()
    client.prepare_index_params.return_value = MagicMock()
    return client


@pytest.fixture()
def mock_async_milvus_client():
    """Return an AsyncMock that stands in for AsyncMilvusClient."""
    return AsyncMock()


@pytest.fixture()
def vector_store(mock_milvus_client, mock_async_milvus_client):
    """MilvusVectorStore with all Milvus I/O mocked out."""
    with _patched_milvus(mock_milvus_client, mock_async_milvus_client):
        return MilvusVectorStore(
            uri="./test.db",
            collection_name=COLLECTION,
            dim=DIM,
            overwrite=False,
            use_async_client=True,
        )


@pytest.fixture()
def sparse_vector_store(mock_milvus_client, mock_async_milvus_client):
    """MilvusVectorStore with sparse embedding enabled."""
    with _patched_milvus(mock_milvus_client, mock_async_milvus_client):
        return MilvusVectorStore(
            uri="./test.db",
            collection_name=COLLECTION,
            dim=DIM,
            overwrite=False,
            enable_sparse=True,
            sparse_embedding_function=FakeSparseEmbeddingFunction(),
            use_async_client=True,
        )


# ---------------------------------------------------------------------------
# Tests: add()
# ---------------------------------------------------------------------------


def test_add_returns_node_ids(vector_store, mock_milvus_client):
    """add() should return the node_id for every node inserted."""
    nodes = [_make_node() for _ in range(3)]
    mock_milvus_client.insert.return_value = {"insert_count": 3}

    ids = vector_store.add(nodes)

    assert len(ids) == 3
    assert set(ids) == {n.node_id for n in nodes}


def test_add_calls_insert_once_per_batch(vector_store, mock_milvus_client):
    """add() should call client.insert at least once."""
    nodes = [_make_node() for _ in range(5)]
    mock_milvus_client.insert.return_value = {}

    vector_store.add(nodes)

    assert mock_milvus_client.insert.called


def test_add_upsert_mode(mock_milvus_client, mock_async_milvus_client):
    """When upsert_mode=True, client.upsert should be called instead of insert."""
    with _patched_milvus(mock_milvus_client, mock_async_milvus_client):
        store = MilvusVectorStore(
            uri="./test.db",
            collection_name=COLLECTION,
            dim=DIM,
            upsert_mode=True,
            use_async_client=False,
        )

    nodes = [_make_node()]
    mock_milvus_client.upsert.return_value = {}

    store.add(nodes)

    mock_milvus_client.upsert.assert_called_once()
    mock_milvus_client.insert.assert_not_called()


def test_add_with_sparse_embeddings(sparse_vector_store, mock_milvus_client):
    """add() with enable_sparse should include sparse vectors."""
    nodes = [_make_node()]
    mock_milvus_client.insert.return_value = {}

    ids = sparse_vector_store.add(nodes)

    assert len(ids) == 1
    # Verify the inserted data contained the sparse_embedding field
    call_args = mock_milvus_client.insert.call_args
    inserted_batch = call_args[0][1]
    assert "sparse_embedding" in inserted_batch[0]


# ---------------------------------------------------------------------------
# Tests: async_add()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_add_returns_node_ids(vector_store, mock_async_milvus_client):
    """async_add() should return the same ids as add()."""
    nodes = [_make_node() for _ in range(2)]
    mock_async_milvus_client.insert.return_value = {}

    ids = await vector_store.async_add(nodes)

    assert set(ids) == {n.node_id for n in nodes}


# ---------------------------------------------------------------------------
# Tests: delete()
# ---------------------------------------------------------------------------


def test_delete_removes_by_ref_doc_id(vector_store, mock_milvus_client):
    """delete() should query by doc_id then call client.delete with primary keys."""
    ref_doc_id = str(uuid.uuid4())
    mock_milvus_client.query.return_value = [{"id": "pk-1"}, {"id": "pk-2"}]

    vector_store.delete(ref_doc_id)

    mock_milvus_client.delete.assert_called_once()
    delete_kwargs = mock_milvus_client.delete.call_args
    assert "pk-1" in delete_kwargs[1]["pks"]
    assert "pk-2" in delete_kwargs[1]["pks"]


def test_delete_noop_when_not_found(vector_store, mock_milvus_client):
    """delete() should not call client.delete when no entries are found."""
    mock_milvus_client.query.return_value = []

    vector_store.delete("nonexistent-id")

    mock_milvus_client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: query() - default (dense) mode
# ---------------------------------------------------------------------------


def _setup_search_result(mock_client, node: TextNode, distance: float = 0.9):
    """Configure mock client.search to return one hit (plain dict, no MagicMock)."""
    mock_client.search.return_value = [[_make_milvus_hit(node, distance)]]


def test_query_default_returns_results(vector_store, mock_milvus_client):
    """query() in DEFAULT mode should call client.search and return results."""
    node = _make_node("The quick brown fox")
    _setup_search_result(mock_milvus_client, node)

    q = VectorStoreQuery(
        query_embedding=[0.1] * DIM,
        similarity_top_k=1,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    result = vector_store.query(q)

    mock_milvus_client.search.assert_called_once()
    assert result.ids is not None
    assert len(result.ids) == 1
    assert result.ids[0] == node.node_id


def test_query_with_metadata_filter(vector_store, mock_milvus_client):
    """query() should pass filter expression when MetadataFilters are provided."""
    node = _make_node()
    _setup_search_result(mock_milvus_client, node)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="test", operator=FilterOperator.EQ)
        ]
    )
    q = VectorStoreQuery(
        query_embedding=[0.1] * DIM,
        similarity_top_k=1,
        filters=filters,
    )
    vector_store.query(q)

    call_kwargs = mock_milvus_client.search.call_args[1]
    assert "category" in call_kwargs["filter"]
    assert "test" in call_kwargs["filter"]


def test_query_with_compound_and_filter(vector_store, mock_milvus_client):
    """query() compound AND filter should include both conditions in expression."""
    node = _make_node()
    _setup_search_result(mock_milvus_client, node)

    filters = MetadataFilters(
        condition=FilterCondition.AND,
        filters=[
            MetadataFilter(key="category", value="test", operator=FilterOperator.EQ),
            MetadataFilter(key="status", value="active", operator=FilterOperator.EQ),
        ],
    )
    q = VectorStoreQuery(
        query_embedding=[0.1] * DIM,
        similarity_top_k=1,
        filters=filters,
    )
    vector_store.query(q)

    filter_expr = mock_milvus_client.search.call_args[1]["filter"]
    assert "category" in filter_expr
    assert "status" in filter_expr
    assert "and" in filter_expr.lower()


def test_query_unsupported_mode_raises(vector_store):
    """query() should raise ValueError for unsupported modes."""
    q = VectorStoreQuery(
        query_embedding=[0.1] * DIM,
        similarity_top_k=1,
        mode=VectorStoreQueryMode.LINEAR_REGRESSION,
    )
    with pytest.raises(ValueError, match="does not support"):
        vector_store.query(q)


def test_query_sparse_mode_raises_without_sparse_enabled(vector_store):
    """SPARSE mode should raise ValueError when enable_sparse is False."""
    q = VectorStoreQuery(
        query_str="hello",
        similarity_top_k=1,
        mode=VectorStoreQueryMode.SPARSE,
    )
    with pytest.raises(ValueError, match="enable_sparse"):
        vector_store.query(q)


# ---------------------------------------------------------------------------
# Tests: sparse / text_search query
# ---------------------------------------------------------------------------


def test_sparse_query_calls_sparse_search(sparse_vector_store, mock_milvus_client):
    """SPARSE mode should use the sparse embedding field for search."""
    node = _make_node()
    _setup_search_result(mock_milvus_client, node)

    q = VectorStoreQuery(
        query_str="hello world",
        similarity_top_k=1,
        mode=VectorStoreQueryMode.SPARSE,
    )
    result = sparse_vector_store.query(q)

    call_kwargs = mock_milvus_client.search.call_args[1]
    assert call_kwargs["anns_field"] == "sparse_embedding"
    assert result.ids is not None


# ---------------------------------------------------------------------------
# Tests: async query (aquery)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aquery_default_returns_results(vector_store, mock_async_milvus_client):
    """aquery() in DEFAULT mode should call aclient.search and return results."""
    node = _make_node("async search result")
    mock_async_milvus_client.search.return_value = [[_make_milvus_hit(node, 0.85)]]

    q = VectorStoreQuery(
        query_embedding=[0.1] * DIM,
        similarity_top_k=1,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    result = await vector_store.aquery(q)

    mock_async_milvus_client.search.assert_called_once()
    assert result.ids is not None
    assert len(result.ids) == 1


@pytest.mark.asyncio
async def test_aquery_sparse_uses_async_encode_queries(
    sparse_vector_store, mock_async_milvus_client
):
    """
    Regression test for fix_milvus_sparse_search:
    _async_sparse_search must await async_encode_queries (not call sync encode_queries).
    """
    node = _make_node("sparse async")
    mock_async_milvus_client.search.return_value = [[_make_milvus_hit(node, 0.7)]]

    fn = sparse_vector_store.sparse_embedding_function
    # AsyncMock wrapping the real sync method lets us assert it was awaited.
    spy = AsyncMock(side_effect=fn.encode_queries)

    q = VectorStoreQuery(
        query_str="sparse hello",
        similarity_top_k=1,
        mode=VectorStoreQueryMode.SPARSE,
    )
    with patch.object(fn, "async_encode_queries", spy):
        result = await sparse_vector_store.aquery(q)

    spy.assert_awaited_once()
    assert result.ids is not None


# ---------------------------------------------------------------------------
# Tests: delete_nodes()
# ---------------------------------------------------------------------------


def test_delete_nodes_by_node_ids(vector_store, mock_milvus_client):
    """delete_nodes() with node_ids should build a filter and call client.delete."""
    node_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    mock_milvus_client.delete.return_value = {}

    vector_store.delete_nodes(node_ids=node_ids)

    mock_milvus_client.delete.assert_called_once()


def test_delete_nodes_by_metadata_filter(vector_store, mock_milvus_client):
    """delete_nodes() with MetadataFilters should pass filter expression."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="old", operator=FilterOperator.EQ)
        ]
    )
    mock_milvus_client.delete.return_value = {}

    vector_store.delete_nodes(filters=filters)

    mock_milvus_client.delete.assert_called_once()
    call_kwargs = mock_milvus_client.delete.call_args[1]
    assert "category" in call_kwargs["filter"]


# ---------------------------------------------------------------------------
# Tests: clear()
# ---------------------------------------------------------------------------


def test_clear_drops_collection(vector_store, mock_milvus_client):
    """clear() should drop the collection."""
    vector_store.clear()

    mock_milvus_client.drop_collection.assert_called_once_with(COLLECTION)


@pytest.mark.asyncio
async def test_aclear_drops_collection(vector_store, mock_async_milvus_client):
    """aclear() should drop the collection via async client."""
    await vector_store.aclear()

    mock_async_milvus_client.drop_collection.assert_called_once_with(COLLECTION)


# ---------------------------------------------------------------------------
# Tests: get_nodes()
# ---------------------------------------------------------------------------


def test_get_nodes_by_ids(vector_store, mock_milvus_client):
    """get_nodes() by node_ids should query and return parsed nodes."""
    node = _make_node("fetched node")
    mock_milvus_client.query.return_value = [
        {
            "id": node.node_id,
            "text": node.text,
            "doc_id": node.node_id,
            "_node_type": "TextNode",
            "_node_content": node.json(),
        }
    ]

    result = vector_store.get_nodes(node_ids=[node.node_id])

    assert len(result) == 1
    assert result[0].text == node.text


def test_get_nodes_requires_ids_or_filters(vector_store):
    """get_nodes() should raise ValueError when neither node_ids nor filters given."""
    with pytest.raises(ValueError, match="Either node_ids or filters must be provided"):
        vector_store.get_nodes()


def test_get_nodes_rejects_both_ids_and_filters(vector_store):
    """get_nodes() should raise ValueError when both node_ids and filters are given."""
    filters = MetadataFilters(
        filters=[MetadataFilter(key="k", value="v", operator=FilterOperator.EQ)]
    )
    with pytest.raises(ValueError, match="Only one of"):
        vector_store.get_nodes(node_ids=["id-1"], filters=filters)
