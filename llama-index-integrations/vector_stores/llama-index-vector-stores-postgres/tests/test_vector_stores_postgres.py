import pytest
from sqlalchemy.dialects import postgresql

from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.postgres import PGVectorStore


def _compile_pg_filter(filters: MetadataFilters) -> str:
    store = PGVectorStore.model_construct()
    clause = store._recursively_apply_filters(filters)
    return str(
        clause.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_class():
    names_of_base_classes = [b.__name__ for b in PGVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_ne_filter_includes_null_and_missing_keys():
    sql_str = _compile_pg_filter(
        MetadataFilters(
            filters=[MetadataFilter(key="k", value="x", operator=FilterOperator.NE)]
        )
    )
    assert sql_str == "(metadata_->>'k' IS NULL OR metadata_->>'k' != 'x')"

    sql_num = _compile_pg_filter(
        MetadataFilters(
            filters=[MetadataFilter(key="k", value=5, operator=FilterOperator.NE)]
        )
    )
    assert sql_num == "(metadata_->>'k' IS NULL OR (metadata_->>'k')::float != 5.0)"


def test_nin_filter_includes_null_and_missing_keys():
    sql = _compile_pg_filter(
        MetadataFilters(
            filters=[
                MetadataFilter(key="k", value=["x", "y"], operator=FilterOperator.NIN)
            ]
        )
    )
    assert sql == "(metadata_->>'k' IS NULL OR metadata_->>'k' NOT IN ('x', 'y'))"


def test_eq_and_in_filters_do_not_match_null():
    sql_eq = _compile_pg_filter(
        MetadataFilters(
            filters=[MetadataFilter(key="k", value="x", operator=FilterOperator.EQ)]
        )
    )
    assert sql_eq == "metadata_->>'k' = 'x'"

    sql_in = _compile_pg_filter(
        MetadataFilters(
            filters=[MetadataFilter(key="k", value=["x"], operator=FilterOperator.IN)]
        )
    )
    assert sql_in == "metadata_->>'k' IN ('x')"


def test_not_filter_condition_single_multiple_and_nested():
    sql_single = _compile_pg_filter(
        MetadataFilters(
            condition=FilterCondition.NOT,
            filters=[MetadataFilter(key="k", value="x", operator=FilterOperator.EQ)],
        )
    )
    assert sql_single == "NOT coalesce(metadata_->>'k' = 'x', false)"

    sql_multi = _compile_pg_filter(
        MetadataFilters(
            condition=FilterCondition.NOT,
            filters=[
                MetadataFilter(key="k", value="x", operator=FilterOperator.EQ),
                MetadataFilter(key="k", value="y", operator=FilterOperator.EQ),
            ],
        )
    )
    assert (
        sql_multi
        == "NOT coalesce(metadata_->>'k' = 'x' OR metadata_->>'k' = 'y', false)"
    )

    sql_nested = _compile_pg_filter(
        MetadataFilters(
            condition=FilterCondition.NOT,
            filters=[
                MetadataFilters(
                    condition=FilterCondition.AND,
                    filters=[
                        MetadataFilter(key="k", value="x", operator=FilterOperator.EQ),
                        MetadataFilter(key="m", value=1, operator=FilterOperator.EQ),
                    ],
                )
            ],
        )
    )
    assert (
        sql_nested
        == "NOT coalesce(metadata_->>'k' = 'x' AND (metadata_->>'m')::float = 1.0, false)"
    )


def test_invalid_filter_condition_raises():
    store = PGVectorStore.model_construct()
    bad_filters = MetadataFilters.model_construct(
        condition="xor",
        filters=[MetadataFilter(key="k", value="x", operator=FilterOperator.EQ)],
    )
    with pytest.raises(ValueError, match="Invalid condition"):
        store._recursively_apply_filters(bad_filters)
