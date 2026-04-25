from unittest.mock import MagicMock

from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.vector_stores.postgres.base import PGVectorStore as _PGVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in PGVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def _make_store() -> PGVectorStore:
    """Create a minimal PGVectorStore using __new__ so no DB connection is needed."""
    store = _PGVectorStore.__new__(_PGVectorStore)
    # _to_postgres_operator only needs the operator enum, so we delegate to the
    # real implementation; we just avoid touching any DB-related attributes.
    return store


def test_build_filter_clause_string_with_underscores():
    """String values that look like numbers (e.g., '2024_123') must use string
    comparison, not a ::float cast.  Python's float() accepts underscores, but
    PostgreSQL's ::float cast does not, so using float() as the type check
    caused DataError for such values (issue #15962)."""
    store = _make_store()
    f = MetadataFilter(key="file_name", value="2024_123", operator=FilterOperator.EQ)
    clause = store._build_filter_clause(f)
    sql = clause.text
    assert "::float" not in sql, f"string value should not use ::float cast; got: {sql}"
    assert "'2024_123'" in sql


def test_build_filter_clause_int_value():
    """Integer filter values should use the ::float cast for numeric comparison."""
    store = _make_store()
    f = MetadataFilter(key="year", value=2024, operator=FilterOperator.EQ)
    clause = store._build_filter_clause(f)
    sql = clause.text
    assert "::float" in sql, f"int value should use ::float cast; got: {sql}"
    assert "2024" in sql


def test_build_filter_clause_float_value():
    """Float filter values should use the ::float cast."""
    store = _make_store()
    f = MetadataFilter(key="score", value=0.9, operator=FilterOperator.GT)
    clause = store._build_filter_clause(f)
    sql = clause.text
    assert "::float" in sql, f"float value should use ::float cast; got: {sql}"


def test_build_filter_clause_plain_number_string():
    """A string value that happens to be a plain decimal number (e.g., '123')
    should still use string comparison because the user explicitly provided a
    string type."""
    store = _make_store()
    f = MetadataFilter(key="year", value="123", operator=FilterOperator.EQ)
    clause = store._build_filter_clause(f)
    sql = clause.text
    assert "::float" not in sql, f"string '123' should not use ::float cast; got: {sql}"
    assert "'123'" in sql
