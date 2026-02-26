"""Tests for the Turbopuffer vector store integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.turbopuffer import TurbopufferVectorStore
from llama_index.vector_stores.turbopuffer.base import _to_turbopuffer_filter
from turbopuffer.types import NamespaceQueryResponse, Row

_DEFAULT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Helpers to build fake turbopuffer responses
# ---------------------------------------------------------------------------


def _make_row(
    id: str, dist: float, attributes: dict[str, Any]
) -> Row:
    """Build a real turbopuffer Row with the given attributes."""
    return Row.model_validate({"id": id, "$dist": dist, **attributes})


def _make_query_response(
    rows: list[Row] | None = None,
) -> NamespaceQueryResponse:
    """Build a NamespaceQueryResponse with the given rows."""
    return NamespaceQueryResponse.model_construct(rows=rows or [])


@dataclass
class _FakeNamespace:
    """Records write/query/delete_all calls for assertions."""

    write_calls: list[dict[str, Any]] = field(default_factory=list)
    query_result: NamespaceQueryResponse | None = None
    delete_all_called: bool = False

    def write(self, **kwargs: Any) -> None:
        self.write_calls.append(kwargs)

    def query(self, **kwargs: Any) -> NamespaceQueryResponse:
        return self.query_result or _make_query_response()

    def delete_all(self, **kwargs: Any) -> None:
        self.delete_all_called = True


def _make_store(
    fake_ns: _FakeNamespace, batch_size: int = _DEFAULT_BATCH_SIZE
) -> TurbopufferVectorStore:
    """Create a TurbopufferVectorStore with a fake namespace."""
    store = TurbopufferVectorStore.model_construct(
        distance_metric="cosine_distance",
        batch_size=batch_size,
        stores_text=True,
        flat_metadata=True,
    )
    store._namespace = fake_ns
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def node_embeddings() -> list[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa-111",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")
            },
            metadata={"author": "Stephen King", "theme": "Friendship"},
            embedding=[1.0, 0.0, 0.0],
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb-222",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")
            },
            metadata={"director": "Francis Ford Coppola", "theme": "Mafia"},
            embedding=[0.0, 1.0, 0.0],
        ),
        TextNode(
            text="consectetur adipiscing",
            id_="ccc-333",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")
            },
            metadata={"director": "Christopher Nolan"},
            embedding=[0.0, 0.0, 1.0],
        ),
    ]


# ---------------------------------------------------------------------------
# Class hierarchy test
# ---------------------------------------------------------------------------


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in TurbopufferVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


# ---------------------------------------------------------------------------
# Filter transformation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operator, expected_op",
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
def test_filter_transform_operators(
    operator: FilterOperator, expected_op: str
) -> None:
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
    # Single filter should not be wrapped in a condition tuple.
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
            MetadataFilter(
                key="field", value=None, operator=FilterOperator.IS_EMPTY
            )
        ]
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("field", "Eq", None)


def test_filter_transform_all_single() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tags", value=["a"], operator=FilterOperator.ALL)
        ]
    )
    result = _to_turbopuffer_filter(filters)
    assert result == ("tags", "Contains", "a")


def test_filter_transform_all_multiple() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tags", value=["a", "b"], operator=FilterOperator.ALL
            )
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
            MetadataFilter(
                key="author", value="King", operator=FilterOperator.EQ
            ),
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
                    MetadataFilter(
                        key="year", value=2000, operator=FilterOperator.GT
                    ),
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
                    MetadataFilter(
                        key="year", value=1950, operator=FilterOperator.GT
                    ),
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
                    MetadataFilter(
                        key="b", value="2", operator=FilterOperator.EQ
                    ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    result = _to_turbopuffer_filter(filters)
    # The nested group has one filter, so it unwraps to a plain tuple.
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
    # Empty nested group is ignored; single remaining filter unwraps.
    assert result == ("a", "Eq", "1")


# ---------------------------------------------------------------------------
# Add tests
# ---------------------------------------------------------------------------


def test_add(node_embeddings: list[TextNode]) -> None:
    fake_ns = _FakeNamespace()
    store = _make_store(fake_ns)

    ids = store.add(node_embeddings)

    assert ids == ["aaa-111", "bbb-222", "ccc-333"]
    assert len(fake_ns.write_calls) == 1

    call = fake_ns.write_calls[0]
    assert call["distance_metric"] == "cosine_distance"
    rows = list(call["upsert_rows"])
    assert len(rows) == 3
    assert rows[0]["id"] == "aaa-111"
    assert rows[0]["vector"] == [1.0, 0.0, 0.0]
    # Metadata should include _node_content, doc_id, and user metadata.
    assert "_node_content" in rows[0]
    assert "doc_id" in rows[0]
    assert rows[0]["author"] == "Stephen King"


def test_add_batching(node_embeddings: list[TextNode]) -> None:
    fake_ns = _FakeNamespace()
    store = _make_store(fake_ns, batch_size=2)

    ids = store.add(node_embeddings)

    assert ids == ["aaa-111", "bbb-222", "ccc-333"]
    # 3 nodes with batch_size=2 should result in 2 write calls.
    assert len(fake_ns.write_calls) == 2
    rows_1 = list(fake_ns.write_calls[0]["upsert_rows"])
    rows_2 = list(fake_ns.write_calls[1]["upsert_rows"])
    assert len(rows_1) == 2
    assert len(rows_2) == 1


def test_add_empty() -> None:
    fake_ns = _FakeNamespace()
    store = _make_store(fake_ns)

    ids = store.add([])

    assert ids == []
    assert len(fake_ns.write_calls) == 0


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


def test_delete() -> None:
    fake_ns = _FakeNamespace()
    store = _make_store(fake_ns)

    store.delete("my-doc-id")

    assert len(fake_ns.write_calls) == 1
    assert fake_ns.write_calls[0]["delete_by_filter"] == (
        "doc_id",
        "Eq",
        "my-doc-id",
    )


def test_delete_nodes_by_ids() -> None:
    fake_ns = _FakeNamespace()
    store = _make_store(fake_ns)

    store.delete_nodes(node_ids=["aaa-111", "bbb-222"])

    assert len(fake_ns.write_calls) == 1
    assert fake_ns.write_calls[0]["deletes"] == ["aaa-111", "bbb-222"]


def test_delete_nodes_by_filters() -> None:
    fake_ns = _FakeNamespace()
    store = _make_store(fake_ns)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="author", value="King", operator=FilterOperator.EQ
            )
        ]
    )
    store.delete_nodes(filters=filters)

    assert len(fake_ns.write_calls) == 1
    assert fake_ns.write_calls[0]["delete_by_filter"] == (
        "author",
        "Eq",
        "King",
    )


def test_clear() -> None:
    fake_ns = _FakeNamespace()
    store = _make_store(fake_ns)

    store.clear()

    assert fake_ns.delete_all_called


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


def _make_node_content_json(
    text: str, node_id: str, metadata: dict[str, Any]
) -> str:
    """Build a _node_content JSON blob like node_to_metadata_dict creates."""
    node = TextNode(text=text, id_=node_id, metadata=metadata, embedding=None)
    node_dict = node.model_dump(mode="json")
    node_dict["embedding"] = None
    return json.dumps(node_dict, ensure_ascii=False)


def test_query_basic() -> None:
    node_content = _make_node_content_json(
        "hello world", "aaa-111", {"k": "v"}
    )
    fake_ns = _FakeNamespace(
        query_result=_make_query_response(
            rows=[
                _make_row(
                    id="aaa-111",
                    dist=0.2,
                    attributes={
                        "_node_content": node_content,
                        "_node_type": "TextNode",
                        "doc_id": "test-0",
                        "document_id": "test-0",
                        "ref_doc_id": "test-0",
                        "k": "v",
                    },
                )
            ]
        )
    )
    store = _make_store(fake_ns)

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1
    )
    result = store.query(query)

    assert result.ids == ["aaa-111"]
    assert len(result.similarities) == 1
    # cosine_distance: similarity = 1 - 0.2 = 0.8
    assert abs(result.similarities[0] - 0.8) < 1e-6
    assert result.nodes[0].get_content() == "hello world"


def test_query_euclidean() -> None:
    node_content = _make_node_content_json("test", "x-1", {})
    fake_ns = _FakeNamespace(
        query_result=_make_query_response(
            rows=[
                _make_row(
                    id="x-1",
                    dist=4.0,
                    attributes={
                        "_node_content": node_content,
                        "_node_type": "TextNode",
                        "doc_id": "None",
                        "document_id": "None",
                        "ref_doc_id": "None",
                    },
                )
            ]
        )
    )
    store = _make_store(fake_ns)
    store.distance_metric = "euclidean_squared"

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1
    )
    result = store.query(query)

    # euclidean_squared: similarity = 1 / (1 + 4) = 0.2
    assert abs(result.similarities[0] - 0.2) < 1e-6


def test_query_with_filters() -> None:
    fake_ns = _FakeNamespace(query_result=_make_query_response())
    store = _make_store(fake_ns)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="author", value="King", operator=FilterOperator.EQ
            )
        ]
    )
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0], similarity_top_k=5, filters=filters
    )
    # Should not raise — filters are transformed and passed through.
    store.query(query)


def test_query_no_embedding() -> None:
    fake_ns = _FakeNamespace()
    store = _make_store(fake_ns)

    query = VectorStoreQuery(query_embedding=None, similarity_top_k=5)
    result = store.query(query)

    assert result.nodes == []
    assert result.similarities == []
    assert result.ids == []


def test_query_fallback_on_bad_metadata() -> None:
    """When _node_content is missing/invalid, query returns a basic TextNode."""
    fake_ns = _FakeNamespace(
        query_result=_make_query_response(
            rows=[
                _make_row(
                    id="bad-1",
                    dist=0.1,
                    attributes={"custom_key": "custom_val"},
                )
            ]
        )
    )
    store = _make_store(fake_ns)

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0], similarity_top_k=1
    )
    result = store.query(query)

    assert result.ids == ["bad-1"]
    assert len(result.nodes) == 1
    node = result.nodes[0]
    assert isinstance(node, TextNode)
    assert node.metadata.get("custom_key") == "custom_val"


# ---------------------------------------------------------------------------
# Integration tests (require TURBOPUFFER_API_KEY)
# ---------------------------------------------------------------------------

skip_integration = pytest.mark.skipif(
    not os.environ.get("TURBOPUFFER_API_KEY"),
    reason="TURBOPUFFER_API_KEY not set",
)


@skip_integration
def test_e2e_add_and_query(node_embeddings: list[TextNode]) -> None:
    import uuid

    from turbopuffer import Turbopuffer

    client = Turbopuffer()
    ns_name = f"llama-test-{uuid.uuid4().hex[:8]}"
    store = TurbopufferVectorStore(namespace=client.namespace(ns_name))

    try:
        ids = store.add(node_embeddings)
        assert len(ids) == 3

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0], similarity_top_k=1
        )
        result = store.query(query)
        assert result.nodes
        assert result.nodes[0].get_content() == "lorem ipsum"
    finally:
        store.clear()


@skip_integration
def test_e2e_delete(node_embeddings: list[TextNode]) -> None:
    import uuid

    from turbopuffer import Turbopuffer

    client = Turbopuffer()
    ns_name = f"llama-test-{uuid.uuid4().hex[:8]}"
    store = TurbopufferVectorStore(namespace=client.namespace(ns_name))

    try:
        store.add(node_embeddings)
        store.delete_nodes(node_ids=["aaa-111"])

        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0], similarity_top_k=10
        )
        result = store.query(query)
        result_ids = result.ids or []
        assert "aaa-111" not in result_ids
    finally:
        store.clear()


@skip_integration
def test_e2e_metadata_filters(node_embeddings: list[TextNode]) -> None:
    import uuid

    from turbopuffer import Turbopuffer

    client = Turbopuffer()
    ns_name = f"llama-test-{uuid.uuid4().hex[:8]}"
    store = TurbopufferVectorStore(namespace=client.namespace(ns_name))

    try:
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
        assert all(
            n.metadata.get("author") == "Stephen King" for n in result.nodes
        )
    finally:
        store.clear()
