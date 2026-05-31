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


def test_ne_matches_nodes_missing_the_key() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.NE, value="news")
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # No node has the "missing" key, so all are trivially not equal to "news".
    assert fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_nin_matches_nodes_missing_the_key() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="missing",
                operator=FilterOperator.NIN,
                value=["news", "blog"],
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # No node has the "missing" key, so it is in none of the listed values.
    assert fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_eq_still_excludes_nodes_missing_the_key() -> None:
    # Regression guard: the missing-key fix for NE/NIN must not change the
    # behavior of positive operators like EQ.
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.EQ, value="news")
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert not fn("n1")
    assert not fn("n2")
    assert not fn("n3")


def test_ne_on_missing_key_agrees_with_not_eq_condition() -> None:
    # `key != value` must produce the same result as `NOT(key == value)` for a
    # node that does not have the key.
    ne_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.NE, value="news")
        ]
    )
    not_eq_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.EQ, value="news")
        ],
        condition=FilterCondition.NOT,
    )
    ne_fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, ne_filters)
    not_eq_fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, not_eq_filters)

    for node_id in _METADATA_BY_ID:
        assert ne_fn(node_id) == not_eq_fn(node_id)


def test_and_condition_with_ne_on_missing_key_combines_with_present_filters() -> None:
    # NE on a missing key should match every node, so the AND result is driven
    # entirely by the second filter on a key that is actually present.
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.NE, value="news"),
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="b"),
        ],
        condition=FilterCondition.AND,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert not fn("n1")
    assert fn("n2")
    assert not fn("n3")


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
