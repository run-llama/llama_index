"""Testing solr vector store utils."""

from typing import Union

import pytest

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.solr.query_utils import (
    recursively_unpack_filters,
)


@pytest.mark.parametrize(
    ("input_filters", "expected_output"),
    [
        (MetadataFilters(filters=[]), []),
        (
            MetadataFilters(
                filters=[MetadataFilter(key="f1", value=1, operator=FilterOperator.GT)],
            ),
            ["((f1:{1 TO *]))"],
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="f1", value=1, operator=FilterOperator.GT),
                    MetadataFilter(key="f2", value=2, operator=FilterOperator.GTE),
                    MetadataFilter(key="f3", value=3, operator=FilterOperator.LT),
                    MetadataFilter(key="f4", value=4, operator=FilterOperator.LTE),
                ],
            ),
            ["((f1:{1 TO *]) AND (f2:[2 TO *]) AND (f3:[* TO 3}) AND (f4:[* TO 4]))"],
        ),
        (
            MetadataFilters(
                filters=[MetadataFilter(key="f1", value=1, operator=FilterOperator.GT)],
                condition=FilterCondition.AND,
            ),
            ["((f1:{1 TO *]))"],
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="f5", value=5, operator=FilterOperator.EQ),
                    MetadataFilter(key="f6", value=6, operator=FilterOperator.NE),
                    MetadataFilter(key="f7", value="v1", operator=FilterOperator.IN),
                    MetadataFilter(
                        key="f8", value=["v1", "v2"], operator=FilterOperator.IN
                    ),
                ],
                condition=FilterCondition.AND,
            ),
            ["((f5:5) AND (-(f6:6)) AND (f7:v1) AND (f8:(v1 OR v2)))"],
        ),
        (
            MetadataFilters(
                filters=[MetadataFilter(key="f1", value=1, operator=FilterOperator.GT)],
                condition=FilterCondition.OR,
            ),
            ["((f1:{1 TO *]))"],
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="f9", value="v1", operator=FilterOperator.NIN),
                    MetadataFilter(
                        key="f10", value=["v1", "v2"], operator=FilterOperator.NIN
                    ),
                    MetadataFilter(
                        key="f11", value="v3", operator=FilterOperator.TEXT_MATCH
                    ),
                ],
                condition=FilterCondition.OR,
            ),
            ['((-f9:v1) OR (-f10:(v1 OR v2)) OR (f11:"v3"))'],
        ),
        (
            MetadataFilters(
                filters=[MetadataFilter(key="f1", value=1, operator=FilterOperator.GT)],
                condition=FilterCondition.NOT,
            ),
            ["(NOT ((f1:{1 TO *])))"],
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="f1", value=1, operator=FilterOperator.GT),
                    MetadataFilter(key="f2", value=2, operator=FilterOperator.GTE),
                    MetadataFilter(key="f3", value=3, operator=FilterOperator.LT),
                    MetadataFilter(key="f4", value=4, operator=FilterOperator.LTE),
                ],
                condition=FilterCondition.NOT,
            ),
            [
                "(NOT ((f1:{1 TO *]) AND (f2:[2 TO *]) AND (f3:[* TO 3}) AND (f4:[* TO 4])))"
            ],
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilters(
                        filters=[
                            MetadataFilter(
                                key="field1", operator=FilterOperator.GT, value=10
                            ),
                            MetadataFilter(
                                key="field2", operator=FilterOperator.LT, value=20
                            ),
                            MetadataFilter(
                                key="field2", operator=FilterOperator.NE, value=15
                            ),
                        ],
                        condition=FilterCondition.AND,
                    ),
                    MetadataFilter(
                        key="field3", operator=FilterOperator.EQ, value="value3"
                    ),
                ],
                condition=FilterCondition.OR,
            ),
            [
                "(((field1:{10 TO *]) AND (field2:[* TO 20}) AND (-(field2:15))) OR (field3:value3))"
            ],
        ),
        (
            MetadataFilters(
                filters=[MetadataFilter(key="f1", value=1, operator=FilterOperator.GT)],
                condition=None,
            ),
            ["(f1:{1 TO *])"],
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="f1", value=1, operator=FilterOperator.GT),
                    MetadataFilter(key="f2", value=2, operator=FilterOperator.GTE),
                    MetadataFilter(key="f3", value=3, operator=FilterOperator.LT),
                    MetadataFilter(key="f4", value=4, operator=FilterOperator.LTE),
                ],
                condition=None,
            ),
            ["(f1:{1 TO *])", "(f2:[2 TO *])", "(f3:[* TO 3})", "(f4:[* TO 4])"],
        ),
    ],
    ids=[
        "Empty subfilters list",
        "Implicit AND of one filter",
        "Implicit AND of multiple filters",
        "Explicit AND of one filter",
        "Explicit AND of multiple filters",
        "OR of one filter",
        "OR of multiple filters",
        "NOT of one filter",
        "NOT of multiple filters (implicit AND)",
        "Nested MetadataFilters",
        "Condition=None for one filter",
        "Condition=None for multiple filters (multiple strings returned)",
    ],
)
def test_recursively_unpack_filters_valid_inputs(
    input_filters: MetadataFilters,
    expected_output: list[str],
) -> None:
    actual_output = recursively_unpack_filters(input_filters)

    assert actual_output == expected_output


@pytest.mark.parametrize(
    ("input_operator", "input_value", "error_match"),
    [
        # value type does not matter
        (FilterOperator.CONTAINS, "some_string", "Disallowed operator used in filter"),
        # value type matters
        (
            FilterOperator.TEXT_MATCH,
            10,
            "Query filter uses a non-string with the 'TEXT_MATCH'",
        ),
        (
            FilterOperator.TEXT_MATCH,
            2.0,
            "Query filter uses a non-string with the 'TEXT_MATCH'",
        ),
    ],
    ids=[
        "Unsupported operator: contains",
        "text_match operator with int",
        "text_match operator with float",
    ],
)
def test_recursively_unpack_filters_invalid_operators(
    input_operator: FilterOperator,
    input_value: str,
    error_match: str,
) -> None:
    input_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="f1", value=1, operator=FilterOperator.GT),
            MetadataFilter(key="f2", value=input_value, operator=input_operator),
        ],
        condition=FilterCondition.AND,
    )

    with pytest.raises(ValueError, match=error_match):
        _ = recursively_unpack_filters(input_filters)


@pytest.mark.parametrize(
    "input_operator",
    [
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
        FilterOperator.EQ,
        FilterOperator.NE,
        FilterOperator.TEXT_MATCH,
    ],
)
@pytest.mark.parametrize(
    "input_value",
    [["string", "list"], [1, 2], [1.0, 2.0]],
    ids=["string list", "int list", "float list"],
)
def test_recursively_unpack_filters_invalid_list_value_with_non_list_operator(
    input_operator: FilterOperator,
    input_value: Union[list[str], list[int], list[float]],
) -> None:
    input_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="f1", value=1, operator=FilterOperator.GT),
            MetadataFilter(key="f2", value=input_value, operator=input_operator),
        ],
        condition=FilterCondition.AND,
    )

    with pytest.raises(
        ValueError, match="Query filter uses a list value for an incompatible operator"
    ):
        _ = recursively_unpack_filters(input_filters)


@pytest.mark.parametrize(
    ("operator", "value", "expected_warning"),
    [
        (FilterOperator.ANY, "single_value", "treating as 'EQ' operator"),
        (FilterOperator.ALL, "single_value", "treating as 'EQ' operator"),
        (FilterOperator.IN, "single_value", "treating as 'EQ' operator"),
        (FilterOperator.NIN, "single_value", "treating as 'NE' operator"),
    ],
    ids=[
        "ANY with non-list",
        "ALL with non-list",
        "IN with non-list",
        "NIN with non-list",
    ],
)
def test_recursively_unpack_filters_warnings(
    operator: FilterOperator, value: str, expected_warning: str, caplog
) -> None:
    input_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="f1", value=value, operator=operator),
        ],
        condition=FilterCondition.AND,
    )

    with caplog.at_level("WARNING"):
        result = recursively_unpack_filters(input_filters)

    assert len(result) == 1
    assert expected_warning in caplog.text


def test_recursively_unpack_filters_no_condition_warning(caplog) -> None:
    input_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="f1", value=1, operator=FilterOperator.GT),
            MetadataFilter(key="f2", value=2, operator=FilterOperator.GT),
        ],
        condition=None,
    )

    with caplog.at_level("WARNING"):
        result = recursively_unpack_filters(input_filters)

    assert len(result) == 2
    assert (
        "No filter condition specified, sub-filters will be returned unlinked"
        in caplog.text
    )


def test_any_and_in_list_equivalence() -> None:
    """ANY and IN with list values should compile to equivalent Solr queries."""
    any_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tags", value=["v1", "v2"], operator=FilterOperator.ANY)
        ]
    )
    in_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tags", value=["v1", "v2"], operator=FilterOperator.IN)
        ]
    )

    any_output = recursively_unpack_filters(any_filters)
    in_output = recursively_unpack_filters(in_filters)

    assert any_output == in_output == ["((tags:(v1 OR v2)))"]


def test_all_list_and_semantics() -> None:
    """ALL with a list should AND the values."""
    all_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tags", value=["v1", "v2"], operator=FilterOperator.ALL)
        ]
    )
    output = recursively_unpack_filters(all_filters)
    assert output == ["((tags:(v1 AND v2)))"]


def test_all_any_in_fallbacks_warnings(caplog) -> None:
    """Non-list value for ALL/ANY/IN should fallback to EQ with warning."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="f_all", value="x", operator=FilterOperator.ALL),
            MetadataFilter(key="f_any", value="y", operator=FilterOperator.ANY),
            MetadataFilter(key="f_in", value="z", operator=FilterOperator.IN),
        ],
        condition=FilterCondition.AND,
    )
    with caplog.at_level("WARNING"):
        out = recursively_unpack_filters(filters)

    # Expect one combined AND group
    assert len(out) == 1
    assert "treating as 'EQ' operator" in caplog.text
    # Basic shape check
    assert "(f_all:x)" in out[0]
    assert "(f_any:y)" in out[0]
    assert "(f_in:z)" in out[0]
