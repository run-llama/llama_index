import math

from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter
from llama_index.vector_stores.postgres import PGVectorStore


def _store() -> PGVectorStore:
    # Construction does not connect; _build_filter_clause is pure SQL.
    return PGVectorStore.from_params(
        database="unused",
        host="localhost",
        password="unused",
        port=5432,
        user="unused",
        table_name="t",
        embed_dim=4,
        perform_setup=False,
    )


def _clause_sql(value: float, operator: FilterOperator) -> str:
    clause = _store()._build_filter_clause(
        MetadataFilter(key="f", value=value, operator=operator)
    )
    return clause.text


def test_finite_float_is_interpolated_as_a_number() -> None:
    sql = _clause_sql(1.5, FilterOperator.EQ)
    assert sql == "(metadata_->>'f')::float = 1.5"


def test_inf_is_quoted_as_postgres_infinity() -> None:
    sql = _clause_sql(math.inf, FilterOperator.LT)
    assert sql == "(metadata_->>'f')::float < 'Infinity'::float"


def test_neg_inf_is_quoted_as_postgres_neg_infinity() -> None:
    sql = _clause_sql(-math.inf, FilterOperator.GT)
    assert sql == "(metadata_->>'f')::float > '-Infinity'::float"


def test_nan_is_quoted_as_postgres_nan() -> None:
    sql = _clause_sql(float("nan"), FilterOperator.EQ)
    assert sql == "(metadata_->>'f')::float = 'NaN'::float"
