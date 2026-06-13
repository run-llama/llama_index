"""
Unit tests for SQL-injection safety of metadata-filter and doc_id handling.

These tests do not require a live Db2 instance: they exercise the WHERE-clause
builder directly and assert that filter values use ibm-db ``?`` parameter
markers (data) rather than inlined SQL, and that the JSON path key is
restricted to a safe identifier charset.

Regression guard for the sibling vulnerability of CVE-2025-1793
(GHSA-v3c8-3pr6-gr7p), completing the parameterization started in
run-llama/llama_index#18316 for the two omitted stores (mariadb, db2).
"""

import pytest

from llama_index.vector_stores.db2.base import DB2LlamaVS

# Build a bare instance without connecting to a DB.
_vs = object.__new__(DB2LlamaVS)
object.__setattr__(_vs, "metadata_column", "metadata")

INJECTION = "category' OR 1=1 -- "


class _Filter:
    def __init__(self, key, value):
        self.key = key
        self.value = value


def test_filter_value_uses_parameter_marker() -> None:
    where_str, bind_values = _vs._append_meta_filter_condition(
        None, [_Filter("author", "Stephen King")]
    )
    assert where_str == "JSON_VALUE(metadata, '$.author') = ?"
    assert bind_values == ["Stephen King"]


def test_injection_in_value_does_not_change_structure() -> None:
    where_str, bind_values = _vs._append_meta_filter_condition(
        None, [_Filter("author", INJECTION)]
    )
    # Attacker payload becomes a bound value, not SQL.
    assert where_str == "JSON_VALUE(metadata, '$.author') = ?"
    assert bind_values == [INJECTION]
    assert "OR 1=1" not in where_str


def test_injection_in_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid metadata key format"):
        _vs._append_meta_filter_condition(None, [_Filter(INJECTION, "x")])


def test_multiple_filters_chain_with_markers() -> None:
    where_str, bind_values = _vs._append_meta_filter_condition(
        None, [_Filter("author", "King"), _Filter("theme", "Mafia")]
    )
    assert where_str == (
        "JSON_VALUE(metadata, '$.author') = ? AND "
        "JSON_VALUE(metadata, '$.theme') = ?"
    )
    assert bind_values == ["King", "Mafia"]


def test_existing_where_str_is_preserved() -> None:
    where_str, bind_values = _vs._append_meta_filter_condition(
        "doc_id in (?, ?)", [_Filter("author", "King")]
    )
    assert where_str == ("doc_id in (?, ?) AND JSON_VALUE(metadata, '$.author') = ?")
    assert bind_values == ["King"]
