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
