import pytest

from llama_index.vector_stores.mongodb.pipelines import (
    filters_to_search_filter,
    filters_to_mql,
    fulltext_search_stage,
    vector_search_stage,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


def test_filters_to_search_filter_and_logic() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
            MetadataFilter(key="genre", value="Comedy", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.AND,
    )
    result = filters_to_search_filter(filters)
    assert "must" in result
    # range for year + equals for genre
    assert any(clause.get("range") for clause in result["must"])  # year
    assert any(clause.get("equals") for clause in result["must"])  # genre


def test_filters_to_search_filter_or_logic_with_negatives() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value=["a", "b"], operator=FilterOperator.IN),
            MetadataFilter(key="status", value="archived", operator=FilterOperator.NE),
        ],
        condition=FilterCondition.OR,
    )
    result = filters_to_search_filter(filters)
    assert "should" in result and result["minimumShouldMatch"] == 1
    # NE should land in mustNot
    assert "mustNot" in result and any(
        clause.get("equals") for clause in result["mustNot"]
    )


def test_filters_to_mql_single() -> None:
    filters = MetadataFilters(
        filters=[MetadataFilter(key="rating", value=5, operator=FilterOperator.GTE)]
    )
    mql = filters_to_mql(filters)
    assert mql == {"metadata.rating": {"$gte": 5}}


def test_vector_search_stage_empty_filter_omitted() -> None:
    stage = vector_search_stage(
        query_vector=[0.1, 0.2],
        search_field="embedding",
        index_name="vec_index",
        limit=2,
        filter={},  # empty dict should be ignored
    )
    assert "$vectorSearch" in stage
    assert "filter" not in stage["$vectorSearch"], "Empty filter should not be attached"


def test_fulltext_search_stage_legacy_filter_alias() -> None:
    legacy = {"must": [{"equals": {"path": "metadata.genre", "value": "Drama"}}]}
    pipeline = fulltext_search_stage(
        query="some text",
        search_field="text",
        index_name="search_index",
        operator="text",
        filter=legacy,  # legacy parameter name
    )
    search_stage = pipeline[0]["$search"]
    compound = search_stage["compound"]
    assert any(
        clause.get("equals") and clause["equals"]["value"] == "Drama"
        for clause in compound["must"]
    )


def test_fulltext_search_stage_conflict_raises() -> None:
    legacy = {"must": []}
    with pytest.raises(ValueError):
        fulltext_search_stage(
            query="q",
            search_field="text",
            index_name="idx",
            operator="text",
            filter=legacy,
            search_filter={},
        )


def test_fulltext_search_stage_search_filter_explicit() -> None:
    filt = {"must": [{"equals": {"path": "metadata.lang", "value": "en"}}]}
    pipeline = fulltext_search_stage(
        query="hello",
        search_field="text",
        index_name="search_index",
        operator="text",
        search_filter=filt,
    )
    compound = pipeline[0]["$search"]["compound"]
    assert any(
        clause.get("equals") and clause["equals"]["value"] == "en"
        for clause in compound["must"]
    )
