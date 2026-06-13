"""
Unit tests for SQL-injection safety of metadata-filter handling.

These tests do not require a live MariaDB instance: they exercise the
WHERE-clause builder directly and assert that filter values are emitted as
SQLAlchemy bind parameters (data) rather than inlined SQL, and that the JSON
path key is restricted to a safe identifier charset.

Regression guard for the sibling vulnerability of CVE-2025-1793
(GHSA-v3c8-3pr6-gr7p), completing the parameterization started in
run-llama/llama_index#18316 for the two omitted stores (mariadb, db2).
"""

import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from llama_index.vector_stores.mariadb import MariaDBVectorStore

# Build a bare instance without connecting to a DB.
_vs = object.__new__(MariaDBVectorStore)

INJECTION = "category' OR 1=1 -- "


def test_filter_value_is_bound_parameter() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="author", value="Stephen King", operator=FilterOperator.EQ
            )
        ],
        condition=FilterCondition.AND,
    )
    clause, params = _vs._filters_to_where_clause(filters)

    assert clause == "JSON_VALUE(metadata, '$.author') = :filter_0"
    assert params == {"filter_0": "Stephen King"}


def test_injection_in_value_does_not_change_structure() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="author", value=INJECTION, operator=FilterOperator.EQ)
        ],
        condition=FilterCondition.AND,
    )
    clause, params = _vs._filters_to_where_clause(filters)

    # The attacker payload is captured as a single bound value, not inlined SQL.
    assert clause == "JSON_VALUE(metadata, '$.author') = :filter_0"
    assert params == {"filter_0": INJECTION}
    # The payload must not appear in the SQL text itself.
    assert "OR 1=1" not in clause


def test_injection_in_key_is_rejected() -> None:
    filters = MetadataFilters(
        filters=[MetadataFilter(key=INJECTION, value="x", operator=FilterOperator.EQ)],
        condition=FilterCondition.AND,
    )
    with pytest.raises(ValueError, match="Invalid metadata key format"):
        _vs._filters_to_where_clause(filters)


def test_in_operator_uses_multiple_bind_params() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="director",
                value=["Francis Ford Coppola", "Christopher Nolan"],
                operator=FilterOperator.IN,
            )
        ],
        condition=FilterCondition.AND,
    )
    clause, params = _vs._filters_to_where_clause(filters)

    assert clause == "JSON_VALUE(metadata, '$.director') IN (:filter_0, :filter_1)"
    assert params == {
        "filter_0": "Francis Ford Coppola",
        "filter_1": "Christopher Nolan",
    }


def test_nested_filters_have_unique_param_names() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="theme", value="Mafia", operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="author", value="X", operator=FilterOperator.NE),
                    MetadataFilter(key="pages", value=10, operator=FilterOperator.LTE),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    clause, params = _vs._filters_to_where_clause(filters)

    assert clause == (
        "JSON_VALUE(metadata, '$.theme') = :filter_0 AND "
        "(JSON_VALUE(metadata, '$.author') != :filter_1 OR "
        "JSON_VALUE(metadata, '$.pages') <= :filter_2)"
    )
    assert params == {"filter_0": "Mafia", "filter_1": "X", "filter_2": 10}
