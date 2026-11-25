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


_BOOL_METADATA_BY_ID: Dict[str, Dict[str, Any]] = {
    "bool_false": {"stage": False},
    "bool_true": {"stage": True},
    "int_0": {"stage": 0},
    "int_1": {"stage": 1},
    "float_1": {"stage": 1.0},
}


def _bool_filter_fn(operator: FilterOperator, value: Any) -> Any:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="stage", operator=operator, value=value)]
    )
    return build_metadata_filter_fn(_BOOL_METADATA_BY_ID.__getitem__, filters)


def test_bool_metadata_never_matches_numeric_equality_filters() -> None:
    # Bool is a subclass of int in Python, but filters must not conflate them:
    # MetadataFilter values are strict ints/floats/strs, so a bool metadata value
    # never equals a numeric filter value, matching backends such as Chroma.
    # See https://github.com/run-llama/llama_index/issues/24452
    fn = _bool_filter_fn(FilterOperator.EQ, 0)
    assert not fn("bool_false")  # False == 0 must not match
    assert not fn("bool_true")
    assert fn("int_0")

    fn = _bool_filter_fn(FilterOperator.EQ, 1)
    assert not fn("bool_true")  # True == 1 must not match
    assert not fn("bool_false")
    assert fn("int_1")
    assert fn("float_1")  # numeric 1.0 == 1 still matches


def test_bool_metadata_always_matches_numeric_inequality_filters() -> None:
    fn = _bool_filter_fn(FilterOperator.NE, 0)
    assert fn("bool_false")  # False != 0 must match
    assert fn("bool_true")
    assert not fn("int_0")
    assert fn("int_1")

    fn = _bool_filter_fn(FilterOperator.NE, 1)
    assert fn("bool_true")  # True != 1 must match
    assert fn("bool_false")
    assert not fn("int_1")
    assert not fn("float_1")


def test_bool_metadata_ordering_comparisons_are_refused() -> None:
    # Ordering between bool metadata and numeric filter values never matches.
    for operator in (
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    ):
        fn = _bool_filter_fn(operator, 0)
        assert not fn("bool_false"), operator
        assert not fn("bool_true"), operator

    # Numeric ordering keeps working as before.
    assert _bool_filter_fn(FilterOperator.GT, 0)("int_1")
    assert _bool_filter_fn(FilterOperator.GTE, 0)("int_0")
    assert _bool_filter_fn(FilterOperator.LT, 1)("int_0")
    assert _bool_filter_fn(FilterOperator.LTE, 1)("float_1")


def test_bool_metadata_membership_filters_stay_type_strict() -> None:
    # IN: bool metadata never matches numeric list elements
    fn = _bool_filter_fn(FilterOperator.IN, [0])
    assert not fn("bool_false")  # False in [0] must not match
    assert not fn("bool_true")
    assert fn("int_0")

    fn = _bool_filter_fn(FilterOperator.IN, [1, 2])
    assert not fn("bool_true")  # True in [1, 2] must not match
    assert fn("int_1")
    assert fn("float_1")

    # NIN: negation of the type-aware IN semantics
    fn = _bool_filter_fn(FilterOperator.NIN, [0])
    assert fn("bool_false")
    assert fn("bool_true")
    assert not fn("int_0")

    fn = _bool_filter_fn(FilterOperator.NIN, [1, 2])
    assert fn("bool_true")
    assert not fn("int_1")
    assert not fn("float_1")


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
