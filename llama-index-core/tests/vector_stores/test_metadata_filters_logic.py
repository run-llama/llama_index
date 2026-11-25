from typing import Any, Dict

import pytest

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.utils import build_metadata_filter_fn


_METADATA_BY_ID: Dict[str, Dict[str, Any]] = {
    "n1": {
        "rank": "a",
        "weight": 1.0,
        "tags": ["alpha", "beta"],
        "identifier": "ABC123",
    },
    "n2": {
        "rank": "b",
        "weight": 2.0,
        "tags": ["beta"],
        "identifier": "abc456",
    },
    "n3": {
        "rank": "c",
        "weight": 3.0,
        "tags": [],
        "identifier": "XYZ789",
    },
}


def test_and_condition_matches_expected_nodes() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.NE, value="a"),
            MetadataFilter(
                key="weight",
                operator=FilterOperator.GTE,
                value=2.0,
            ),
        ],
        condition=FilterCondition.AND,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert not fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_none_condition_defaults_to_and() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.NE, value="a"),
            MetadataFilter(key="weight", operator=FilterOperator.GTE, value=2.0),
        ],
        condition=None,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert not fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_or_condition_matches_expected_nodes() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="c"),
        ],
        condition=FilterCondition.OR,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert fn("n1")
    assert not fn("n2")
    assert fn("n3")


def test_not_condition_negates_filters() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
        ],
        condition=FilterCondition.NOT,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert not fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_is_empty_matches_missing_and_empty_values() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="missing",
                operator=FilterOperator.IS_EMPTY,
                value=None,
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # All nodes are missing the "missing" key, so they should all match.
    assert fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_ne_and_nin_match_missing_metadata_values() -> None:
    ne_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.NE, value="a"),
        ]
    )
    nin_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.NIN, value=["a"]),
        ]
    )

    ne_fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, ne_filters)
    nin_fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, nin_filters)

    # Missing metadata values are not equal to, and not contained in, the
    # filtered value. This keeps NE/NIN consistent with NOT(EQ)/NOT(IN).
    assert ne_fn("n1")
    assert nin_fn("n1")


def test_text_match_insensitive_uses_case_insensitive_search() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="identifier",
                operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
                value="abc",
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # Both identifiers contain "abc" ignoring case.
    assert fn("n1")
    assert fn("n2")
    assert not fn("n3")


def test_text_match_requires_string_values() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="weight",
                operator=FilterOperator.TEXT_MATCH,
                value="1",
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # Non-string metadata values should raise a TypeError when used with TEXT_MATCH.
    with pytest.raises(TypeError):
        fn("n1")


def test_text_match_insensitive_requires_string_values() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="weight",
                operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
                value="1",
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # Non-string metadata values should raise a TypeError when used with TEXT_MATCH_INSENSITIVE.
    with pytest.raises(TypeError):
        fn("n1")


def test_nested_metadata_filters_raise_error() -> None:
    inner = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
        ]
    )
    outer = MetadataFilters(
        filters=[inner],
        condition=FilterCondition.AND,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, outer)

    # Nested MetadataFilters are not supported and should raise an error
    # when the filter function is evaluated.
    with pytest.raises(ValueError, match="Nested MetadataFilters are not supported"):
        fn("n1")

_BOOL_INT_METADATA: Dict[str, Dict[str, Any]] = {
    "b1": {"stage": False},
    "b2": {"stage": 0},
    "b3": {"stage": True},
    "b4": {"stage": 1},
}


def _matches(operator: FilterOperator, value: Any) -> list:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="stage", operator=operator, value=value)]
    )
    fn = build_metadata_filter_fn(_BOOL_INT_METADATA.__getitem__, filters)
    return [fn(node_id) for node_id in _BOOL_INT_METADATA]


def test_bool_and_int_are_distinct_for_eq_and_ne() -> None:
    # Chroma and the other backends keep bool and int apart (#24450, #24452);
    # Python's False == 0 / True == 1 must not make SimpleVectorStore diverge.
    assert _matches(FilterOperator.EQ, 0) == [False, True, False, False]
    assert _matches(FilterOperator.EQ, 1) == [False, False, False, True]
    assert _matches(FilterOperator.NE, 0) == [True, False, True, True]
    assert _matches(FilterOperator.GT, 0) == [False, False, False, True]


_HETEROGENEOUS_METADATA: Dict[str, Dict[str, Any]] = {
    "h1": {"score": "hello"},
    "h2": {"score": 5},
}


def _matches_score(operator: FilterOperator, value: Any) -> list:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="score", operator=operator, value=value)]
    )
    fn = build_metadata_filter_fn(_HETEROGENEOUS_METADATA.__getitem__, filters)
    return [fn(node_id) for node_id in _HETEROGENEOUS_METADATA]


def test_range_filters_skip_incomparable_metadata_instead_of_raising() -> None:
    # A string under a numeric range filter must not raise TypeError and abort
    # the query (#24451).
    assert _matches_score(FilterOperator.LT, 0) == [False, False]
    assert _matches_score(FilterOperator.GT, 0) == [False, True]
    assert _matches_score(FilterOperator.GTE, 5) == [False, True]

