from typing import List, Optional, Tuple, Union

from google.cloud import bigquery
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)


def build_where_clause_and_params(
    node_ids: Optional[List[str]] = None,
    filters: Optional[MetadataFilters] = None,
) -> Tuple[
    str, List[Union[bigquery.ScalarQueryParameter, bigquery.ArrayQueryParameter]]
]:
    """
    Construct a parameterized SQL WHERE clause and corresponding query parameters.

    The clause is built from the provided node IDs and metadata filters. Parameters
    are returned separately to support safe, parameterized queries in BigQuery,
    helping to prevent SQL injection. See:
    https://cloud.google.com/bigquery/docs/parameterized-queries

    If both `node_ids` and `filters` are provided, the resulting WHERE clause
    combines conditions using AND logic.

    Args:
        node_ids: A list of node IDs to include in the filter.
        filters: Metadata filters to apply to the query.

    Returns:
        A tuple (where_clause, query_params), where where_clause is a SQL WHERE clause string,
        and query_params is a list of query parameters to bind to the query.

    """
    query_params: List[
        Union[bigquery.ScalarQueryParameter, bigquery.ArrayQueryParameter]
    ] = []
    conditions: List[str] = []

    if filters:
        filter_where_clause, filter_query_params = (
            _recursive_build_where_clause_from_filters(filters)
        )
        conditions.append(filter_where_clause)
        query_params.extend(filter_query_params)

    if node_ids:
        conditions.append("node_id IN UNNEST(@node_ids)")
        query_params.append(
            bigquery.ArrayQueryParameter(
                name="node_ids", array_type="STRING", values=node_ids
            )
        )

    # if both `node_ids` and `filters` are provided, both criteria should be considered
    where_clause = " AND ".join(conditions)

    return where_clause, query_params


def _recursive_build_where_clause_from_filters(
    meta_filters: MetadataFilters,
) -> Tuple[
    str, List[Union[bigquery.ScalarQueryParameter, bigquery.ArrayQueryParameter]]
]:
    """
    Recursively construct a SQL WHERE clause and corresponding query parameters

    The provided MetadataFilters object may contain nested filter groups. This function
    traverses them recursively to build a complete WHERE clause and the associated
    query parameters for use in a parameterized BigQuery query.

    Args:
        meta_filters: A potentially nested MetadataFilters filter object

    Returns:
        A tuple (where_clause, query_params), where where_clause is a parameterized SQL WHERE clause string,
        and query_params is a list of query parameters to bind to the query.

    """
    filters_list: List[str] = []
    query_params: List[
        Union[bigquery.ScalarQueryParameter, bigquery.ArrayQueryParameter]
    ] = []

    for filter_ in meta_filters.filters:
        clause, params = (
            _recursive_build_where_clause_from_filters(filter_)
            if isinstance(filter_, MetadataFilters)
            else _build_filter_clause(filter_)
        )

        filters_list.append(clause)
        query_params.extend(params)

    condition = f" {meta_filters.condition.value.upper()} "
    filters_ = f"({condition.join(filters_list)})"

    return filters_, query_params


def _build_filter_clause(
    filter_: MetadataFilter,
) -> Tuple[
    str, List[Union[bigquery.ScalarQueryParameter, bigquery.ArrayQueryParameter]]
]:
    field = filter_.key
    operator = filter_.operator
    value = filter_.value

    if operator == FilterOperator.IS_EMPTY:
        clause = f"JSON_TYPE(JSON_QUERY(metadata, '$.\"{field}\"')) = 'null'"
        params = []
    elif operator == FilterOperator.IN or operator == FilterOperator.NIN:
        bigquery_operator = _llama_to_bigquery_operator(operator)
        clause = (
            f"SAFE.JSON_VALUE(metadata, '$.\"{field}\"') {bigquery_operator} UNNEST(?)"
        )
        params = [
            bigquery.ArrayQueryParameter(name=None, array_type="STRING", values=value)
        ]
    elif operator == FilterOperator.TEXT_MATCH:
        bigquery_operator = _llama_to_bigquery_operator(operator)
        clause = (
            f"SAFE.JSON_VALUE(metadata, '$.\"{field}\"') {bigquery_operator} '{value}'"
        )
        params = [bigquery.ScalarQueryParameter(name=None, type_="STRING", value=value)]
    elif operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
        bigquery_operator = _llama_to_bigquery_operator(operator)
        clause = f"LOWER(SAFE.JSON_VALUE(metadata, '$.\"{field}\"')) {bigquery_operator} LOWER('{value}')"
        params = [bigquery.ScalarQueryParameter(name=None, type_="STRING", value=value)]
    else:
        bigquery_operator = _llama_to_bigquery_operator(operator)
        clause = f"SAFE.JSON_VALUE(metadata, '$.\"{field}\"') {bigquery_operator} ?"
        params = [bigquery.ScalarQueryParameter(name=None, type_="STRING", value=value)]

    return clause, params


def _llama_to_bigquery_operator(operator: FilterOperator) -> str:
    operator_map = {
        FilterOperator.EQ: "=",
        FilterOperator.GT: ">",
        FilterOperator.LT: "<",
        FilterOperator.NE: "!=",
        FilterOperator.GTE: ">=",
        FilterOperator.LTE: "<=",
        FilterOperator.IN: "IN",
        FilterOperator.NIN: "NOT IN",
        FilterOperator.TEXT_MATCH: "LIKE",
        FilterOperator.TEXT_MATCH_INSENSITIVE: "LIKE",
        FilterOperator.IS_EMPTY: "IS NULL",
    }

    try:
        return operator_map[operator]
    except KeyError:
        raise ValueError(
            f"Invalid operator `{operator.value}` is not a supported BigQuery operator."
        )
