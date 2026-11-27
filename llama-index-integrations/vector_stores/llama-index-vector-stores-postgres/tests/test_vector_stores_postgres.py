import math

from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
)
from llama_index.vector_stores.postgres import PGVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in PGVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_build_filter_clause_non_finite_floats():
    store = PGVectorStore.model_construct()

    clause_inf = store._build_filter_clause(
        MetadataFilter(key="score", value=math.inf, operator=FilterOperator.EQ)
    )
    assert "'Infinity'::float" in str(clause_inf)

    clause_neg_inf = store._build_filter_clause(
        MetadataFilter(key="score", value=-math.inf, operator=FilterOperator.LT)
    )
    assert "'-Infinity'::float" in str(clause_neg_inf)

    clause_nan = store._build_filter_clause(
        MetadataFilter(key="score", value=math.nan, operator=FilterOperator.NE)
    )
    assert "'NaN'::float" in str(clause_nan)

    clause_finite = store._build_filter_clause(
        MetadataFilter(key="score", value=1.5, operator=FilterOperator.GT)
    )
    assert "::float > 1.5" in str(clause_finite)
