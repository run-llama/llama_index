"""
Regression tests for SQL++ injection in
`_convert_llamaindex_filters_to_sql` (see issue #22314).

These are pure unit tests against a module-level function and require no
live Couchbase connection.
"""
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.couchbase.base import (
    _convert_llamaindex_filters_to_sql,
)

MALICIOUS_VALUE = "' OR 1=1 UNION SELECT d.* FROM sensitive_bucket d --"


def test_eq_filter_does_not_interpolate_value_into_sql() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value=MALICIOUS_VALUE, operator=FilterOperator.EQ
            )
        ]
    )

    sql, params = _convert_llamaindex_filters_to_sql(filters, "metadata")

    # The malicious payload must never appear in the generated SQL string.
    assert MALICIOUS_VALUE not in sql
    assert "'" not in sql
    # It must instead be carried as a bound named parameter.
    assert sql == "d.metadata.category = $metadata_filter_0"
    assert params == {"metadata_filter_0": MALICIOUS_VALUE}


def test_ne_gt_lt_filters_use_named_parameters() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="status", value="banned", operator=FilterOperator.NE),
            MetadataFilter(key="score", value=10, operator=FilterOperator.GT),
            MetadataFilter(key="score", value=100, operator=FilterOperator.LTE),
        ],
        condition="and",
    )

    sql, params = _convert_llamaindex_filters_to_sql(filters, "metadata")

    assert sql == (
        "d.metadata.status != $metadata_filter_0 AND "
        "d.metadata.score > $metadata_filter_1 AND "
        "d.metadata.score <= $metadata_filter_2"
    )
    assert params == {
        "metadata_filter_0": "banned",
        "metadata_filter_1": 10,
        "metadata_filter_2": 100,
    }


def test_in_filter_binds_list_as_single_named_parameter() -> None:
    malicious_list_item = "a', 'b') OR 1=1 --"
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tag",
                value=["safe", malicious_list_item],
                operator=FilterOperator.IN,
            )
        ]
    )

    sql, params = _convert_llamaindex_filters_to_sql(filters, "metadata")

    assert malicious_list_item not in sql
    assert sql == "d.metadata.tag IN $metadata_filter_0"
    assert params == {"metadata_filter_0": ["safe", malicious_list_item]}


def test_in_filter_requires_list_value() -> None:
    import pytest

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tag", value="not-a-list", operator=FilterOperator.IN
            )
        ]
    )

    with pytest.raises(ValueError, match="expects a list value"):
        _convert_llamaindex_filters_to_sql(filters, "metadata")


def test_nested_filters_produce_unique_parameter_names() -> None:
    inner = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=MALICIOUS_VALUE, operator=FilterOperator.EQ),
            MetadataFilter(key="b", value="x", operator=FilterOperator.EQ),
        ],
        condition="or",
    )
    outer = MetadataFilters(
        filters=[
            MetadataFilter(key="c", value=1, operator=FilterOperator.GTE),
            inner,
        ],
        condition="and",
    )

    sql, params = _convert_llamaindex_filters_to_sql(outer, "metadata")

    # Three total leaf filters -> three unique parameter names, no reuse.
    assert len(params) == 3
    assert MALICIOUS_VALUE not in sql
    assert sql == (
        "d.metadata.c >= $metadata_filter_0 AND "
        "(d.metadata.a = $metadata_filter_1 OR d.metadata.b = $metadata_filter_2)"
    )


def test_empty_filters_return_empty_sql_and_params() -> None:
    sql, params = _convert_llamaindex_filters_to_sql(None, "metadata")
    assert sql == ""
    assert params == {}
