import pytest

from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
)
from llama_index.vector_stores.postgres import PGVectorStore

COMPARISON_OPERATORS = [
    (FilterOperator.EQ, "="),
    (FilterOperator.NE, "!="),
    (FilterOperator.GT, ">"),
    (FilterOperator.GTE, ">="),
    (FilterOperator.LT, "<"),
    (FilterOperator.LTE, "<="),
]


def _filter_sql(filter_: MetadataFilter) -> str:
    store = PGVectorStore.__new__(PGVectorStore)
    return store._build_filter_clause(filter_).text


def test_class():
    names_of_base_classes = [b.__name__ for b in PGVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.mark.parametrize(("operator", "sql_operator"), COMPARISON_OPERATORS)
@pytest.mark.parametrize("value", ["2024_123", "123", "0.5", "1e5", "nan", "inf"])
def test_build_filter_clause_numeric_like_string_is_text_comparison(
    value: str, operator: FilterOperator, sql_operator: str
) -> None:
    sql = _filter_sql(MetadataFilter(key="k", value=value, operator=operator))
    assert sql == f"metadata_->>'k' {sql_operator} '{value}'"


@pytest.mark.parametrize(("operator", "sql_operator"), COMPARISON_OPERATORS)
@pytest.mark.parametrize(
    ("value", "sql_value"), [(2024, "2024.0"), (-3, "-3.0"), (0.9, "0.9")]
)
def test_build_filter_clause_numeric_value_casts_to_float(
    value, sql_value: str, operator: FilterOperator, sql_operator: str
) -> None:
    sql = _filter_sql(MetadataFilter(key="k", value=value, operator=operator))
    assert sql == f"(metadata_->>'k')::float {sql_operator} {sql_value}"


def test_build_filter_clause_bool_is_not_cast_to_float() -> None:
    # StrictInt rejects bool at validation, so bypass it to cover the guard.
    filter_ = MetadataFilter.model_construct(
        key="k", value=True, operator=FilterOperator.EQ
    )
    assert "::float" not in _filter_sql(filter_)


@pytest.mark.parametrize(
    ("operator", "sql_operator"),
    [(FilterOperator.IN, "IN"), (FilterOperator.NIN, "NOT IN")],
)
def test_build_filter_clause_in_nin_unchanged(
    operator: FilterOperator, sql_operator: str
) -> None:
    sql = _filter_sql(
        MetadataFilter(key="k", value=["2024_123", "123"], operator=operator)
    )
    assert sql == f"metadata_->>'k' {sql_operator} ('2024_123', '123')"
