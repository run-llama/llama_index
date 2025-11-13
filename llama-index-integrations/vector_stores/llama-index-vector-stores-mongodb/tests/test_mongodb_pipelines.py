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


def test_filters_to_mql_is_empty_single() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="country", value=None, operator=FilterOperator.IS_EMPTY)
        ]
    )
    mql = filters_to_mql(filters)
    assert "$or" in mql, "IS_EMPTY single filter should expand to $or grouping"
    clauses = mql["$or"]
    # Expect three branches: $exists False, None equality, empty array equality
    keys = [next(iter(c.keys())) for c in clauses]
    assert all(k == "metadata.country" for k in keys)
    assert any(clause["metadata.country"] == [] for clause in clauses)
    assert any(clause["metadata.country"] is None for clause in clauses)
    assert any(
        clause["metadata.country"].get("$exists") is False
        for clause in clauses
        if isinstance(clause.get("metadata.country"), dict)
    )


def test_filters_to_mql_is_empty_or_with_in() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="country",
                value=["FR", "CA"],
                operator=FilterOperator.IN,
            ),
            MetadataFilter(key="country", value=None, operator=FilterOperator.IS_EMPTY),
        ],
        condition=FilterCondition.OR,
    )
    mql = filters_to_mql(filters)
    assert "$or" in mql
    # Ensure first IN clause present
    assert any(
        "metadata.country" in clause
        and isinstance(clause["metadata.country"], dict)
        and clause["metadata.country"].get("$in") == ["FR", "CA"]
        for clause in mql["$or"]
    )
    # Ensure nested $or for IS_EMPTY present
    assert any("$or" in clause for clause in mql["$or"])


def test_filters_to_search_filter_is_empty() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="country", value=None, operator=FilterOperator.IS_EMPTY)
        ]
    )
    search_filter = filters_to_search_filter(filters)
    # AND context for single filter -> expect nested compound in must
    if "must" in search_filter:
        # find nested compound with should or mustNot exists
        assert any(
            clause.get("compound") and (
                clause["compound"].get("should") or clause["compound"].get("mustNot")
            )
            for clause in search_filter["must"]
        ) or any(
            clause.get("equals") and clause["equals"]["value"] in (None, [])
            for clause in search_filter["must"]
        )
    else:
        # OR path (should) -- minimumShouldMatch present
        assert "should" in search_filter
        assert search_filter.get("minimumShouldMatch") == 1
        assert any(
            clause.get("compound") and clause["compound"].get("mustNot")
            for clause in search_filter["should"]
        )
        assert any(
            clause.get("equals") and clause["equals"]["value"] in (None, [])
            for clause in search_filter["should"]
        )
