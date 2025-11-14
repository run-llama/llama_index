import pytest

from llama_index.vector_stores.mongodb.pipelines import (
    filters_to_atlas_search_compound,
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


def test_filters_to_atlas_search_compound_and_logic() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
            MetadataFilter(key="genre", value="Comedy", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.AND,
    )
    result = filters_to_atlas_search_compound(filters)
    assert "must" in result
    # range for year + equals for genre
    assert any(clause.get("range") for clause in result["must"])  # year
    assert any(clause.get("equals") for clause in result["must"])  # genre


def test_filters_to_atlas_search_compound_or_logic_with_negatives() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value=["a", "b"], operator=FilterOperator.IN),
            MetadataFilter(key="status", value="archived", operator=FilterOperator.NE),
        ],
        condition=FilterCondition.OR,
    )
    result = filters_to_atlas_search_compound(filters)
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
            MetadataFilter(key="tags", value=None, operator=FilterOperator.IS_EMPTY)
        ]
    )
    mql = filters_to_mql(filters)
    assert "$or" in mql, "IS_EMPTY single filter should expand to $or grouping"
    clauses = mql["$or"]
    # Expect two branches: $exists False, None equality (empty array omitted for $vectorSearch pre-filter)
    keys = [next(iter(c.keys())) for c in clauses]
    assert all(k == "metadata.tags" for k in keys)
    assert any(clause["metadata.tags"] is None for clause in clauses)
    assert any(
        clause["metadata.tags"].get("$exists") is False
        for clause in clauses
        if isinstance(clause.get("metadata.tags"), dict)
    )


def test_filters_to_mql_is_empty_or_with_in() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tags",
                value=["news", "ai"],
                operator=FilterOperator.IN,
            ),
            MetadataFilter(key="tags", value=None, operator=FilterOperator.IS_EMPTY),
        ],
        condition=FilterCondition.OR,
    )
    mql = filters_to_mql(filters)
    assert "$or" in mql
    # Ensure first IN clause present
    assert any(
        "metadata.tags" in clause
        and isinstance(clause["metadata.tags"], dict)
        and clause["metadata.tags"].get("$in") == ["news", "ai"]
        for clause in mql["$or"]
    )
    # Ensure nested $or for IS_EMPTY present (now only two branches: exists False OR null)
    nested_or = [clause for clause in mql["$or"] if "$or" in clause]
    assert nested_or, "Nested $or for IS_EMPTY should be present"
    assert all(len(inner["$or"]) == 2 for inner in nested_or)


def test_filters_to_mql_is_empty_or_with_in_prefixed_key() -> None:
    """Ensure keys already prefixed with `metadata.` are not double-prefixed and expansion is correct."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="metadata.tags", value=["opensource"], operator=FilterOperator.IN
            ),
            MetadataFilter(
                key="metadata.tags", value=None, operator=FilterOperator.IS_EMPTY
            ),
        ],
        condition=FilterCondition.OR,
    )
    mql = filters_to_mql(filters)
    assert "$or" in mql
    # IN clause present without double prefix
    assert any(
        "metadata.tags" in clause
        and isinstance(clause["metadata.tags"], dict)
        and clause["metadata.tags"].get("$in") == ["opensource"]
        for clause in mql["$or"]
    )
    # Nested OR for IS_EMPTY present (two branches: exists False or null)
    nested = [clause for clause in mql["$or"] if "$or" in clause]
    assert nested, "Expected nested $or for IS_EMPTY"
    assert all(len(inner["$or"]) == 2 for inner in nested)
    # No empty array literal branch
    assert all([] not in [sub.get("metadata.tags") for sub in inner["$or"]] for inner in nested)


def test_filters_to_atlas_search_compound_is_empty() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tags", value=None, operator=FilterOperator.IS_EMPTY)
        ]
    )
    search_filter = filters_to_atlas_search_compound(filters)
    # AND context for single filter -> expect nested compound in must
    if "must" in search_filter:
        # find nested compound with should or mustNot exists
        assert any(
            clause.get("compound") and clause["compound"].get("should")
            for clause in search_filter["must"]
        ), "Expected compound should wrapper for IS_EMPTY"
        # The should branches should contain equals null and a compound mustNot exists; no empty array literal
        compound_clause = next(
            clause for clause in search_filter["must"] if clause.get("compound")
        )
        should_branches = compound_clause["compound"]["should"]
        assert any(
            b.get("equals") and b["equals"]["value"] is None for b in should_branches
        )
        assert any(
            b.get("compound") and b["compound"].get("mustNot") for b in should_branches
        )
        assert not any(
            b.get("equals") and b["equals"]["value"] == [] for b in should_branches
        ), "Empty array literal should be omitted"
    else:
        # OR path (should) -- minimumShouldMatch present
        assert "should" in search_filter
        assert search_filter.get("minimumShouldMatch") == 1
        assert any(
            clause.get("compound") and clause["compound"].get("mustNot")
            for clause in search_filter["should"]
        ), "Missing mustNot exists branch"
        assert any(
            clause.get("equals") and clause["equals"]["value"] is None
            for clause in search_filter["should"]
        ), "Missing equals null branch"
        assert not any(
            clause.get("equals") and clause["equals"]["value"] == []
            for clause in search_filter["should"]
        ), "Empty array literal should be omitted"
