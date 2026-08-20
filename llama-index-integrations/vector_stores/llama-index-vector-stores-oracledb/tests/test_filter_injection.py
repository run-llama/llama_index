"""
Unit tests for SQL-injection safety of the table identifier.

These tests do not require a live Oracle DB instance: they exercise the
identifier validation that guards the raw SQL DDL/DML strings (e.g.
``CREATE TABLE {table_name} ...``, ``DELETE FROM {table_name} ...`` and
``DROP TABLE {table_name} PURGE``) and assert that a table name containing SQL
metacharacters is rejected with ``ValueError`` before it can be interpolated
into SQL.

Regression guard for the sibling vulnerability of CVE-2025-1793, completing the
identifier-allowlist hardening introduced for the postgres store in
run-llama/llama_index#18316 for the oracledb store. (Metadata-filter keys are
already allowlisted and filter values already use bind parameters in this
store; this adds the missing table-name guard.)
"""

import pytest

from llama_index.vector_stores.oracledb.base import OraLlamaVS, _validate_identifier

INJECTION = "vectors; DROP TABLE users; --"


def test_validate_identifier_accepts_plain_name() -> None:
    assert _validate_identifier("vectors", "table_name") == "vectors"
    assert _validate_identifier("my_table_1", "table_name") == "my_table_1"


def test_validate_identifier_rejects_injection() -> None:
    with pytest.raises(ValueError, match="Invalid table_name"):
        _validate_identifier(INJECTION, "table_name")


def test_constructor_rejects_malicious_table_name() -> None:
    # Validation runs before any DB connection is attempted, so a bogus client
    # is fine here: the malicious identifier must be rejected first.
    with pytest.raises(ValueError, match="Invalid table_name"):
        OraLlamaVS(_client=object(), table_name=INJECTION)
