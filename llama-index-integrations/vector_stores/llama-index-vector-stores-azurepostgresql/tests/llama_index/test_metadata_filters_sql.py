"""Unit tests for `metadata_filters_to_sql`.

The generated SQL is rendered with `psycopg.sql.Composable.as_string`, so these
tests do not need a PostgreSQL server. They pin the numeric comparison path:
GT/GTE/LT/LTE filters with an `int` or `float` value compare the metadata value
as a float behind a `jsonb_typeof` guard, while every other value keeps the
existing text comparison.
"""

from typing import Any

import pytest

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.azure_postgres.base import metadata_filters_to_sql


def _render(filters: MetadataFilters) -> str:
    return metadata_filters_to_sql(filters).as_string()


def _render_single(key: str, value: Any, operator: FilterOperator) -> str:
    return _render(
        MetadataFilters(
            filters=[MetadataFilter(key=key, value=value, operator=operator)],
            condition="and",
        )
    )


@pytest.mark.parametrize(
    ("operator", "sql_operator"),
    [
        (FilterOperator.GT, ">"),
        (FilterOperator.GTE, ">="),
        (FilterOperator.LT, "<"),
        (FilterOperator.LTE, "<="),
    ],
)
@pytest.mark.parametrize("value", [30, 9.5])
def test_numeric_value_uses_guarded_float_comparison(
    operator: FilterOperator, sql_operator: str, value: float
) -> None:
    """Int and float values compare numerically behind a `jsonb_typeof` guard."""
    rendered = _render_single("price", value, operator)

    assert "jsonb_typeof(metadata -> 'price') = 'number'" in rendered
    assert f"(metadata ->> 'price')::float {sql_operator} {float(value)}" in rendered
    # The uncast text comparison must not be produced for numeric values.
    assert f"(metadata ->> 'price') {sql_operator}" not in rendered


def test_string_value_keeps_text_comparison() -> None:
    """A string value keeps the existing text comparison without a cast."""
    rendered = _render_single("price", "9", FilterOperator.GT)

    assert rendered == "(metadata ->> 'price') > '9'"
    assert "jsonb_typeof" not in rendered
    assert "::float" not in rendered


def test_bool_value_does_not_take_numeric_path() -> None:
    """A boolean value is not a number even though `bool` subclasses `int`."""
    filter_item = MetadataFilter.model_construct(
        key="enabled", value=True, operator=FilterOperator.GT
    )
    rendered = _render(MetadataFilters(filters=[filter_item], condition="and"))

    assert rendered == "(metadata ->> 'enabled') > true"
    assert "jsonb_typeof" not in rendered
    assert "::float" not in rendered


def test_eq_keeps_text_semantics_for_numeric_value() -> None:
    """EQ is unchanged: it still compares the text representation."""
    rendered = _render_single("price", 30, FilterOperator.EQ)

    assert rendered == "metadata ->> 'price' = '30'"
    assert "jsonb_typeof" not in rendered


def test_nested_and_or_filters_compose() -> None:
    """Numeric and string filters compose inside nested AND/OR filters."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="price", value=10, operator=FilterOperator.GT),
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", value="books", operator=FilterOperator.EQ
                    ),
                    MetadataFilter(
                        key="rating", value=4.5, operator=FilterOperator.LTE
                    ),
                ],
                condition="or",
            ),
        ],
        condition="and",
    )

    rendered = _render(filters)

    assert rendered == (
        "(jsonb_typeof(metadata -> 'price') = 'number'"
        " AND (metadata ->> 'price')::float > 10.0)"
        " AND (metadata ->> 'category' = 'books'"
        " OR (jsonb_typeof(metadata -> 'rating') = 'number'"
        " AND (metadata ->> 'rating')::float <= 4.5))"
    )


def test_missing_or_non_numeric_values_are_guarded() -> None:
    """Rows without a numeric value are filtered out instead of raising.

    `jsonb_typeof(metadata -> 'price')` is NULL when the key is missing and
    'string' (or 'null', 'array', ...) for non-numeric values. In both cases
    the guard does not evaluate to TRUE, so PostgreSQL skips the row without
    ever evaluating a text-to-float comparison on it.
    """
    rendered = _render_single("price", 30, FilterOperator.GT)

    assert "jsonb_typeof(metadata -> 'price') = 'number'" in rendered
