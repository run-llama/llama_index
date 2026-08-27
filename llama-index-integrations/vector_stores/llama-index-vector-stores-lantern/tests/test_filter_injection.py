"""
Unit tests for SQL-injection safety of schema/table identifiers.

These tests do not require a live Lantern/Postgres instance: they exercise the
identifier validation that guards the raw SQL DDL/DML strings (e.g.
``CREATE SCHEMA IF NOT EXISTS {schema_name}`` and
``DELETE FROM {schema_name}.data_{table_name}``) and assert that an identifier
containing SQL metacharacters is rejected with ``ValueError`` before it can be
interpolated into SQL.

Regression guard for the sibling vulnerability of CVE-2025-1793, completing the
identifier-allowlist hardening introduced for the postgres store in
run-llama/llama_index#18316 for the lantern store.
"""

import pytest

from llama_index.vector_stores.lantern.base import _validate_identifier

INJECTION = "public; DROP TABLE users; --"


def test_validate_identifier_accepts_plain_name() -> None:
    assert _validate_identifier("public", "schema_name") == "public"
    assert _validate_identifier("my_table_1", "table_name") == "my_table_1"


def test_validate_identifier_rejects_injection() -> None:
    with pytest.raises(ValueError, match="Invalid schema_name"):
        _validate_identifier(INJECTION, "schema_name")


def test_validate_identifier_rejects_dot_qualified() -> None:
    # A dotted name would let an attacker break out of the intended schema.
    with pytest.raises(ValueError, match="Invalid table_name"):
        _validate_identifier("schema.data_x", "table_name")


def test_constructor_rejects_malicious_schema_name() -> None:
    from llama_index.vector_stores.lantern import LanternVectorStore

    with pytest.raises(ValueError, match="Invalid schema_name"):
        LanternVectorStore.from_params(
            database="db",
            host="localhost",
            password="pw",
            port="5432",
            user="user",
            table_name="legit",
            schema_name=INJECTION,
            embed_dim=3,
            perform_setup=False,
        )


def test_constructor_rejects_malicious_table_name() -> None:
    from llama_index.vector_stores.lantern import LanternVectorStore

    with pytest.raises(ValueError, match="Invalid table_name"):
        LanternVectorStore.from_params(
            database="db",
            host="localhost",
            password="pw",
            port="5432",
            user="user",
            table_name="data; DROP TABLE x; --",
            schema_name="public",
            embed_dim=3,
            perform_setup=False,
        )
