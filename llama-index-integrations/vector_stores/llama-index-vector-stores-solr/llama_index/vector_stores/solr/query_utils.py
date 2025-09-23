"""
Utilities for use with the Apache Solr Vector Store integration.
This module provides utility functions for working with Solr in the context of
LlamaIndex vector stores. It includes functions for:
- Query escaping and preprocessing
- Node relationship serialization/deserialization
- Metadata filter conversion to Solr query syntax
- Sparse vector encoding for Solr's delimited term frequency filters
The utilities handle the transformation between LlamaIndex data structures and
Solr-compatible formats, ensuring proper query syntax and data encoding.
"""

import logging
from types import MappingProxyType
from typing import Union

from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.solr.constants import (
    ESCAPE_RULES_GENERIC,
)

logger = logging.getLogger(__name__)


def escape_query_characters(
    value: str, translation_table: MappingProxyType[int, str] = ESCAPE_RULES_GENERIC
) -> str:
    """
    Escape query characters in order to prevent from user query injection.
    Reference: `Standard Query Parser <https://solr.apache.org/guide/solr/latest/query-guide/standard-query-parser.html#escaping-special-characters>`_

    Args:
        value: The input text to be escaped.
        translation_table: The translation table to use for escaping.

    Returns:
        The input text escaped appropriately for use with Solr.

    """
    return value.translate(translation_table)


def _handle_metadata_filter(subfilter: Union[MetadataFilter, ExactMatchFilter]) -> str:
    """
    Convert a single metadata filter to Solr query string format.
    Handles various filter operators (EQ, NE, GT, LT, IN, NIN, etc.) and converts them
    to appropriate Solr query syntax. Special handling is provided for list values
    with IN/NIN/ALL/ANY operators.

    Args:
        subfilter: A metadata filter or exact match filter to convert.

    Returns:
        A Solr-compatible query string for the filter.

    Raises:
        ValueError: If an unsupported operator is used or if list values are used
            with incompatible operators.

    """
    key = subfilter.key
    value = subfilter.value
    op = subfilter.operator

    # 1. List-handling branches (limited operator support on list values)
    if isinstance(value, list):
        if op == FilterOperator.ALL:
            return f"({key}:({' AND '.join(value)}))"
        if op in (FilterOperator.ANY, FilterOperator.IN):
            # ANY (multi-valued field contains any of) and IN (value is a member of set)
            # both reduce to an OR disjunction in Solr.
            return f"({key}:({' OR '.join([str(x) for x in value])}))"
        if op == FilterOperator.NIN:
            return f"(-{key}:({' OR '.join([str(x) for x in value])}))"
        # Any other operator combined with a list value is invalid
        raise ValueError(
            "Query filter uses a list value for an incompatible operator, only 'IN', 'NIN', 'ANY' and 'ALL' are supported with lists: "
            f"{subfilter}"
        )

    # 2. Fallbacks for list-only operators when the user supplied a scalar value
    if op in (
        FilterOperator.ALL,
        FilterOperator.ANY,
        FilterOperator.IN,
    ) and not isinstance(value, list):
        logger.warning(
            "Query filter contains '%s' operator for non-list value (type=%s), treating as 'EQ' operator: %s",
            op.value,
            type(value),
            subfilter,
        )
        return f"({key}:{value})"
    if op == FilterOperator.NIN and not isinstance(value, list):
        logger.warning(
            "Query filter contains 'NIN' operator for non-list value (type=%s), treating as 'NE' operator: %s",
            type(value),
            subfilter,
        )
        return f"(-{key}:{value})"

    # 3. Scalar operator handling (value is str/int/float/etc.)
    if op == FilterOperator.GT:
        return f"({key}:{{{value} TO *])"
    if op == FilterOperator.GTE:
        return f"({key}:[{value} TO *])"
    if op == FilterOperator.LT:
        return f"({key}:[* TO {value}}})"
    if op == FilterOperator.LTE:
        return f"({key}:[* TO {value}])"
    if op == FilterOperator.EQ:
        return f"({key}:{value})"
    if op == FilterOperator.NE:
        return f"(-({key}:{value}))"
    if op in (FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE):
        if isinstance(value, str):
            # NOTE: Ensure that the field is properly configured for text_match_insensitive in the Solr schema
            return f'({key}:"{value}")'
        if op == FilterOperator.TEXT_MATCH:
            raise ValueError(
                f"Query filter uses a non-string with the 'TEXT_MATCH' operator: {subfilter}"
            )
        # For TEXT_MATCH_INSENSITIVE with non-string, fall through to unknown operator error below.

    # 4. Explicitly disallowed operators
    if op in (FilterOperator.CONTAINS, FilterOperator.IS_EMPTY):
        raise ValueError(f"Disallowed operator used in filter: {subfilter}")

    # 5. Unknown / future operator (parity with original pragma: no cover branch)
    raise ValueError(
        f"Unknown operator used in filter: {subfilter}"
    )  # pragma: no cover


def recursively_unpack_filters(filters: MetadataFilters) -> list[str]:
    """
    Recursively unpack metadata filters to Solr filter query.

    Notes: Solr has issues with complex filters. We have noticed queries to not be
    returning correct results always when you have nested filters. This is not a problem
    with this function as far as we can tell, it is likely an issue with the Solr parser.

    Not all ``llama-index`` filter operations are supported in the optional ``filters``
    attribute of :py:class:`~llama_index.core.vector_stores.VectorStoreQuery`. If any of
    the following filters are passed, an error will be raised:

    * ``contains``
    * ``is_empty``

    See :py:class:`~llama_index.core.vector_stores.types.FilterOperator` for the
    complete list of operators

    Args:
        filters: The set of filters to be converted into Solr-specific query parameters

    Returns:
        A set of filters converted into the Solr-specific query language, linked using the
        relevant query condition (e.g., AND, OR).
        If the input filters do not contain a value for ``condition`` in
        :py:class:`~llama_index.core.vector_stores.types.MetadataFilter` , then the
        sub-filters will be returned without being linked

    Raises:
        ValueError: If an unsupported or unknown filter operator is passed.

    """
    if not filters.filters:
        logger.info("Input MetadataFilters contains no subfilters: %s", filters)
        return []

    # convert all the individual filter statements
    filter_queries: list[str] = []
    for subfilter in filters.filters:
        if isinstance(subfilter, MetadataFilters):
            filter_queries.extend(recursively_unpack_filters(subfilter))
        elif isinstance(subfilter, MetadataFilter):
            filter_queries.append(_handle_metadata_filter(subfilter))
        else:  # pragma: no cover
            raise ValueError(
                f"Unknown subfilter type, type={type(subfilter)}: {subfilter}"
            )

    # combine the filter statements using the appropriate condition
    condition = filters.condition
    if condition == FilterCondition.AND:
        filter_output = [f"({' AND '.join(filter_queries)})"]
    elif condition == FilterCondition.OR:
        filter_output = [f"({' OR '.join(filter_queries)})"]
    elif condition == FilterCondition.NOT:
        filter_output = [f"(NOT ({' AND '.join(filter_queries)}))"]
    elif condition is None:
        logger.warning(
            "No filter condition specified, sub-filters will be returned unlinked"
        )
        filter_output = filter_queries
    else:  # pragma: no cover
        raise ValueError(f"Unknown filter condition: {filters.condition}")

    logger.debug(
        "Converted query filters to Solr filters, input=%s: %s", filters, filter_output
    )
    return filter_output
