"""
Unit tests for SQL-injection safety of metadata-filter keys and the table
identifier.

These tests do not require a live Volcengine RDS MySQL instance: they exercise
the validation that guards the raw SQL strings and assert that an identifier
containing SQL metacharacters is rejected with ``ValueError`` before it can be
interpolated into SQL. They also confirm that benign filter values are emitted
as SQLAlchemy bind parameters (data) rather than inlined SQL.

Regression guard for the sibling vulnerability of CVE-2025-1793, completing the
identifier-allowlist hardening introduced for the postgres store in
run-llama/llama_index#18316 for the volcenginemysql store. Both the JSON-path
metadata key (string-literal sink ``'$.{key}'``) and the table name
(``CREATE TABLE``/``INSERT``/``DELETE``/``SELECT``/``DROP`` sinks) are guarded.
"""

import pytest
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
)

from llama_index.vector_stores.volcengine_mysql import VolcengineMySQLVectorStore
from llama_index.vector_stores.volcengine_mysql.base import _validate_identifier

INJECTION = "category' OR 1=1 -- "
TABLE_INJECTION = "vectors`; DROP TABLE users; -- "

# Build a bare instance without connecting to a DB.
_vs = object.__new__(VolcengineMySQLVectorStore)


def test_validate_identifier_rejects_injection() -> None:
    with pytest.raises(ValueError, match="Invalid table_name"):
        _validate_identifier(TABLE_INJECTION, "table_name")


def test_filter_value_is_bound_parameter() -> None:
    params: dict = {}
    clause = _vs._build_filter_clause(
        MetadataFilter(key="author", value="Stephen King", operator=FilterOperator.EQ),
        params,
        [0],
    )
    # The attacker-controllable value is captured as a bind parameter, not SQL.
    assert clause == (
        "JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.author')) = :filter_param_0"
    )
    assert params == {"filter_param_0": "Stephen King"}


def test_injection_in_value_does_not_change_structure() -> None:
    params: dict = {}
    clause = _vs._build_filter_clause(
        MetadataFilter(key="author", value=INJECTION, operator=FilterOperator.EQ),
        params,
        [0],
    )
    assert clause == (
        "JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.author')) = :filter_param_0"
    )
    assert params == {"filter_param_0": INJECTION}
    assert "OR 1=1" not in clause


def test_injection_in_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid metadata key format"):
        _vs._build_filter_clause(
            MetadataFilter(key=INJECTION, value="x", operator=FilterOperator.EQ),
            {},
            [0],
        )


def test_constructor_rejects_malicious_table_name() -> None:
    with pytest.raises(ValueError, match="Invalid table_name"):
        VolcengineMySQLVectorStore(
            connection_string="mysql+pymysql://user:pwd@localhost:3306/db",
            table_name=TABLE_INJECTION,
            perform_setup=False,
        )
