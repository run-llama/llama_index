"""Tests for the turbopuffer vector store integration."""

from __future__ import annotations

import os
import uuid

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from unittest.mock import MagicMock

from llama_index.vector_stores.turbopuffer import TurbopufferVectorStore
from llama_index.vector_stores.turbopuffer.base import (
    _METADATA_PREFIX,
    _to_turbopuffer_filter,
)

skip_integration = pytest.mark.skipif(
    not os.environ.get("TURBOPUFFER_API_KEY"),
    reason="TURBOPUFFER_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def node_embeddings() -> list[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa-111",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
            metadata={"author": "Stephen King", "theme": "Friendship"},
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb-222",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
            metadata={"director": "Francis Ford Coppola", "theme": "Mafia"},
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="consectetur adipiscing",
            id_="ccc-333",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
            metadata={"director": "Christopher Nolan"},
            embedding=[0.0, 0.0, 1.0],
        ),
    ]


@pytest.fixture()
def store():
    """Create a TurbopufferVectorStore backed by a unique namespace."""
    from turbopuffer import Turbopuffer

    client = Turbopuffer(region="gcp-us-central1")
    ns_name = f"llama-test-{uuid.uuid4().hex[:8]}"
    s = TurbopufferVectorStore(namespace=client.namespace(ns_name))

    # Write and delete a dummy node to ensure the namespace exists on the server.
    dummy = TextNode(text="init", id_="__init__", embedding=[0.0, 0.0, 0.0])
    s.add([dummy])
    s.delete_nodes(node_ids=["__init__"])

    yield s
    s.clear()


@pytest.fixture()
def mock_store() -> TurbopufferVectorStore:
    """TurbopufferVectorStore with a mock namespace for unit tests."""
    mock_ns = MagicMock()
    return TurbopufferVectorStore(namespace=mock_ns)


# ---------------------------------------------------------------------------
# Class hierarchy test
# ---------------------------------------------------------------------------


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in TurbopufferVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


# ---------------------------------------------------------------------------
# Filter transformation tests (pure logic, no API key needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operator", "expected_op"),
    [
        (FilterOperator.EQ, "Eq"),
        (FilterOperator.NE, "NotEq"),
        (FilterOperator.GT, "Gt"),
        (FilterOperator.LT, "Lt"),
        (FilterOperator.GTE, "Gte"),
        (FilterOperator.LTE, "Lte"),
        (FilterOperator.IN, "In"),
        (FilterOperator.NIN, "NotIn"),
        (FilterOperator.CONTAINS, "Contains"),
        (FilterOperator.ANY, "ContainsAny"),
        (FilterOperator.TEXT_MATCH, "Glob"),
        (FilterOperator.TEXT_MATCH_INSENSITIVE, "IGlob"),
    ],
)
def test_filter_transform_operators(operator: FilterOperator, expected_op: str) -> None:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="field", value="val", operator=operator)]
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("field", expected_op, "val")


def test_filter_transform_single_filter() -> None:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="x", value="y", operator=FilterOperator.EQ)]
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("x", "Eq", "y")


def test_filter_transform_empty() -> None:
    filters = MetadataFilters(filters=[])
    result = _to_turbopuffer_filter(filters)
    assert result is None


def test_filter_transform_multiple_and() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            MetadataFilter(key="b", value=2, operator=FilterOperator.GT),
        ],
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("And", [("a", "Eq", "1"), ("b", "Gt", 2)])


def test_filter_transform_is_empty() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="field", value=None, operator=FilterOperator.IS_EMPTY)
        ]
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("field", "Eq", None)


def test_filter_transform_all_single() -> None:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tags", value=["a"], operator=FilterOperator.ALL)]
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("tags", "Contains", "a")


def test_filter_transform_all_multiple() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tags", value=["a", "b"], operator=FilterOperator.ALL)
        ]
    )
    result = _to_turbopuffer_filter(filters)
    assert result == (
        "And",
        [("tags", "Contains", "a"), ("tags", "Contains", "b")],
    )


def test_filter_transform_and_condition() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            MetadataFilter(key="b", value=2, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.AND,
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("And", [("a", "Eq", "1"), ("b", "Gt", 2)])


def test_filter_transform_or_condition() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            MetadataFilter(key="b", value="2", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR,
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("Or", [("a", "Eq", "1"), ("b", "Eq", "2")])


def test_filter_transform_nested_and_with_or() -> None:
    """AND at top level with an OR sub-group."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="author", value="King", operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="theme", value="Mafia", operator=FilterOperator.EQ
                    ),
                    MetadataFilter(
                        key="theme",
                        value="Friendship",
                        operator=FilterOperator.EQ,
                    ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    result = _to_turbopuffer_filter(filters)
    assert result == (
        "And",
        [
            ("author", "Eq", "King"),
            ("Or", [("theme", "Eq", "Mafia"), ("theme", "Eq", "Friendship")]),
        ],
    )


def test_filter_transform_nested_or_with_and() -> None:
    """OR at top level with AND sub-groups."""
    filters = MetadataFilters(
        filters=[
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="author", value="King", operator=FilterOperator.EQ
                    ),
                    MetadataFilter(key="year", value=2000, operator=FilterOperator.GT),
                ],
                condition=FilterCondition.AND,
            ),
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="author",
                        value="Tolkien",
                        operator=FilterOperator.EQ,
                    ),
                    MetadataFilter(key="year", value=1950, operator=FilterOperator.GT),
                ],
                condition=FilterCondition.AND,
            ),
        ],
        condition=FilterCondition.OR,
    )
    result = _to_turbopuffer_filter(filters)
    assert result == (
        "Or",
        [
            ("And", [("author", "Eq", "King"), ("year", "Gt", 2000)]),
            ("And", [("author", "Eq", "Tolkien"), ("year", "Gt", 1950)]),
        ],
    )


def test_filter_transform_nested_single_child() -> None:
    """Nested group with a single filter should unwrap."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="b", value="2", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("And", [("a", "Eq", "1"), ("b", "Eq", "2")])


def test_filter_transform_nested_empty_child() -> None:
    """Nested group with no filters should be skipped."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            MetadataFilters(filters=[], condition=FilterCondition.OR),
        ],
        condition=FilterCondition.AND,
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("a", "Eq", "1")


# ---------------------------------------------------------------------------
# NOT filter tests
# ---------------------------------------------------------------------------


def test_filter_transform_not_single() -> None:
    """NOT with a single filter produces ("Not", filter)."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="status", value="archived", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.NOT,
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("Not", ("status", "Eq", "archived"))


def test_filter_transform_not_multiple() -> None:
    """NOT with multiple filters wraps them in And."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            MetadataFilter(key="b", value="2", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.NOT,
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("Not", ("And", [("a", "Eq", "1"), ("b", "Eq", "2")]))


def test_filter_transform_not_empty() -> None:
    """NOT with no filters returns None."""
    filters = MetadataFilters(filters=[], condition=FilterCondition.NOT)
    result = _to_turbopuffer_filter(filters)
    assert result is None


# ---------------------------------------------------------------------------
# _build_rows / _row_to_node unit tests (no API key needed)
# ---------------------------------------------------------------------------


def test_build_rows_prefixes_reserved_keys(mock_store: TurbopufferVectorStore) -> None:
    """Metadata keys that collide with reserved columns get prefixed."""
    node = TextNode(
        text="hello",
        id_="node-1",
        metadata={"id": "user-id-value", "vector": "user-vector-value"},
        embedding=[1.0, 0.0],
    )
    rows = mock_store._build_rows([node])
    assert len(rows) == 1
    row = rows[0]
    # Reserved keys should be prefixed with _meta_
    assert row[f"{_METADATA_PREFIX}id"] == "user-id-value"
    assert row[f"{_METADATA_PREFIX}vector"] == "user-vector-value"
    # Actual id and vector should be the node's values
    assert row["id"] == "node-1"
    assert row["vector"] == [1.0, 0.0]


def test_row_to_node_round_trip(mock_store: TurbopufferVectorStore) -> None:
    """A node serialized via _build_rows can be deserialized via _row_to_node."""
    node = TextNode(
        text="round trip text",
        id_="rt-1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},
        metadata={"author": "Test Author", "year": 2024},
        embedding=[0.5, 0.5],
    )
    rows = mock_store._build_rows([node])
    row_dict = rows[0]

    # Simulate what turbopuffer returns (no vector in query response).
    row_dict.pop("vector", None)
    row_dict["$dist"] = 0.1

    restored = mock_store._row_to_node(row_dict, str(row_dict["id"]))
    assert restored.node_id == "rt-1"
    assert restored.get_content() == "round trip text"
    assert restored.metadata.get("author") == "Test Author"
    assert restored.metadata.get("year") == 2024


def test_row_to_node_restores_prefixed_metadata(
    mock_store: TurbopufferVectorStore,
) -> None:
    """Metadata that was prefixed during _build_rows gets restored on read."""
    node = TextNode(
        text="prefixed",
        id_="pf-1",
        metadata={"id": "my-custom-id"},
        embedding=[1.0],
    )
    rows = mock_store._build_rows([node])
    row_dict = rows[0]
    row_dict.pop("vector", None)

    restored = mock_store._row_to_node(row_dict, str(row_dict["id"]))
    # The prefixed "_meta_id" should be restored back to "id" in metadata
    assert restored.metadata.get("id") == "my-custom-id"


# ---------------------------------------------------------------------------
# E2E tests (require TURBOPUFFER_API_KEY)
# ---------------------------------------------------------------------------


@skip_integration
def test_add_and_query(
    store: TurbopufferVectorStore, node_embeddings: list[TextNode]
) -> None:
    ids = store.add(node_embeddings)
    assert len(ids) == 3

    query = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1)
    result = store.query(query)
    assert result.nodes
    assert result.nodes[0].get_content() == "lorem ipsum"


@skip_integration
def test_query_no_embedding(store: TurbopufferVectorStore) -> None:
    query = VectorStoreQuery(query_embedding=None, similarity_top_k=5)
    result = store.query(query)

    assert result.nodes == []
    assert result.similarities == []
    assert result.ids == []


@skip_integration
def test_delete(store: TurbopufferVectorStore, node_embeddings: list[TextNode]) -> None:
    store.add(node_embeddings)
    store.delete_nodes(node_ids=["aaa-111"])

    query = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10)
    result = store.query(query)
    result_ids = result.ids or []
    assert "aaa-111" not in result_ids


@skip_integration
def test_metadata_filters(
    store: TurbopufferVectorStore, node_embeddings: list[TextNode]
) -> None:
    store.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="author",
                value="Stephen King",
                operator=FilterOperator.EQ,
            )
        ]
    )
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=10,
        filters=filters,
    )
    result = store.query(query)
    assert result.nodes
    assert all(n.metadata.get("author") == "Stephen King" for n in result.nodes)


# ---------------------------------------------------------------------------
# Relative score fusion tests (pure logic, no API key needed)
# ---------------------------------------------------------------------------


def _make_result(ids: list[str], scores: list[float]) -> VectorStoreQueryResult:
    """Helper to build a VectorStoreQueryResult with stub TextNodes."""
    nodes = [TextNode(text=f"text-{i}", id_=i) for i in ids]
    return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)


def test_rrf_both_empty() -> None:
    empty = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    result = TurbopufferVectorStore._reciprocal_rank_fusion(empty, empty)
    assert result.nodes == []


def test_rrf_sparse_empty() -> None:
    dense = _make_result(["a"], [0.9])
    empty = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    result = TurbopufferVectorStore._reciprocal_rank_fusion(dense, empty)
    assert result.ids == ["a"]


def test_rrf_dense_empty() -> None:
    empty = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    sparse = _make_result(["a"], [0.9])
    result = TurbopufferVectorStore._reciprocal_rank_fusion(empty, sparse)
    assert result.ids == ["a"]


def test_rrf_overlapping_results() -> None:
    dense = _make_result(["a", "b"], [0.8, 0.9])
    sparse = _make_result(["b", "c"], [1.0, 0.5])
    result = TurbopufferVectorStore._reciprocal_rank_fusion(dense, sparse, top_k=10)
    # "b" appears in both lists, so gets RRF score from both ranks
    assert result.ids is not None
    assert result.ids[0] == "b"


def test_rrf_respects_top_k() -> None:
    dense = _make_result(["a", "b", "c"], [1.0, 0.8, 0.6])
    sparse = _make_result(["d", "e"], [1.0, 0.5])
    result = TurbopufferVectorStore._reciprocal_rank_fusion(dense, sparse, top_k=2)
    assert len(result.nodes or []) == 2


def test_rrf_disjoint_results() -> None:
    dense = _make_result(["a"], [1.0])
    sparse = _make_result(["b"], [1.0])
    result = TurbopufferVectorStore._reciprocal_rank_fusion(dense, sparse, top_k=2)
    # Both appear at rank 1 in their respective lists, so equal RRF scores.
    # Either order is valid, just check both are present.
    assert result.ids is not None
    assert set(result.ids) == {"a", "b"}


# ---------------------------------------------------------------------------
# E2E hybrid/BM25 tests (require TURBOPUFFER_API_KEY)
# ---------------------------------------------------------------------------


@skip_integration
def test_text_search(
    store: TurbopufferVectorStore, node_embeddings: list[TextNode]
) -> None:
    store.add(node_embeddings)

    query = VectorStoreQuery(
        query_str="lorem ipsum",
        similarity_top_k=3,
        mode=VectorStoreQueryMode.TEXT_SEARCH,
    )
    result = store.query(query)
    assert result.nodes
    assert result.nodes[0].get_content() == "lorem ipsum"


@skip_integration
def test_hybrid_search(
    store: TurbopufferVectorStore, node_embeddings: list[TextNode]
) -> None:
    store.add(node_embeddings)

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        query_str="lorem ipsum",
        similarity_top_k=3,
        mode=VectorStoreQueryMode.HYBRID,
        alpha=0.5,
    )
    result = store.query(query)
    assert result.nodes
    assert len(result.nodes) > 0
