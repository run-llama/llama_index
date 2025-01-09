from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    MetadataFilter,
)
from llama_index.vector_stores.weaviate.base import _to_weaviate_filter


def test_to_weaviate_filter_with_various_operators():
    filters = MetadataFilters(filters=[MetadataFilter(key="a", value=1)])
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "Equal"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.NE)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "NotEqual"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.GT)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "GreaterThan"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.GTE)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "GreaterThanEqual"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.LT)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "LessThan"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.LTE)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "LessThanEqual"
    assert filter.value == 1

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=None, operator=FilterOperator.IS_EMPTY)]
    )
    filter = _to_weaviate_filter(filters)
    assert filter.target == "a"
    assert filter.operator == "IsNull"
    assert filter.value is True


def test_to_weaviate_filter_with_multiple_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.GTE),
            MetadataFilter(key="a", value=10, operator=FilterOperator.LTE),
        ],
        condition=FilterCondition.AND,
    )
    filter = _to_weaviate_filter(filters)
    assert filter.operator == "And"
    assert len(filter.filters) == 2
    assert filter.filters[0].target == "a"
    assert filter.filters[0].operator == "GreaterThanEqual"
    assert filter.filters[0].value == 1
    assert filter.filters[1].target == "a"
    assert filter.filters[1].operator == "LessThanEqual"
    assert filter.filters[1].value == 10

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.LT),
            MetadataFilter(key="a", value=10, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.OR,
    )
    filter = _to_weaviate_filter(filters)
    assert filter.operator == "Or"
    assert len(filter.filters) == 2
    assert filter.filters[0].target == "a"
    assert filter.filters[0].operator == "LessThan"
    assert filter.filters[0].value == 1
    assert filter.filters[1].target == "a"
    assert filter.filters[1].operator == "GreaterThan"
    assert filter.filters[1].value == 10


def test_to_weaviate_filter_with_nested_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
                    MetadataFilter(key="c", value=3, operator=FilterOperator.GT),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    filter = _to_weaviate_filter(filters)
    assert filter.operator == "And"
    assert len(filter.filters) == 2
    assert filter.filters[0].target == "a"
    assert filter.filters[0].operator == "Equal"
    assert filter.filters[0].value == 1
    assert filter.filters[1].operator == "Or"
    or_filters = filter.filters[1].filters
    assert len(or_filters) == 2
    assert or_filters[0].target == "b"
    assert or_filters[0].operator == "Equal"
    assert or_filters[0].value == 2
    assert or_filters[1].target == "c"
    assert or_filters[1].operator == "GreaterThan"
    assert or_filters[1].value == 3
