"""
Aggregation pipeline components used in Atlas Full-Text, Vector, and Hybrid Search.

"""

from typing import Any, Dict, List, Optional, TypeVar

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilters,
)

MongoDBDocumentType = TypeVar("MongoDBDocumentType", bound=Dict[str, Any])

import logging

logger = logging.getLogger(__name__)


def fulltext_search_stage(
    query: str,
    search_field: str,
    index_name: str,
    operator: str = "text",
    filter: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Full-Text search.

    Args:
        query: Input text to search for
        search_field: Field in Collection that will be searched
        index_name: Atlas Search Index name
        operator: A number of operators are available in the text search stage.

    Returns:
        Dictionary defining the $search

    See Also:
        - MongoDB Full-Text Search <https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search>
        - MongoDB Operators <https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref>

    """
    pipeline = [
        {
            "$search": {
                "index": index_name,
                operator: {"query": query, "path": search_field},
            }
        },
    ]
    if filter:
        pipeline.append({"$match": filter})
    pipeline.append({"$limit": limit})
    return pipeline


def filters_to_mql(
    filters: MetadataFilters, metadata_key: str = "metadata"
) -> Dict[str, Any]:
    """
    Converts Llama-index's MetadataFilters into the MQL expected by $vectorSearch query.

    We are looking for something like

    "filter": {
            "$and": [
                { "metadata.genres": { "$eq": "Comedy" } },
                { "metadata.year": { "$gt": 2010 } }
            ]
    },

    See: See https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter

    Args:
        filters: MetadataFilters object
        metadata_key: The key under which metadata is stored in the document

    Returns:
        MQL version of the filter.

    """
    if filters is None:
        return {}

    def prepare_key(key: str) -> str:
        return (
            f"{metadata_key}.{key}" if not key.startswith(f"{metadata_key}.") else key
        )

    if len(filters.filters) == 1:
        mf = filters.filters[0]
        mql = {
            prepare_key(mf.key): {map_lc_mql_filter_operators(mf.operator): mf.value}
        }
    elif filters.condition == FilterCondition.AND:
        mql = {
            "$and": [
                {
                    prepare_key(mf.key): {
                        map_lc_mql_filter_operators(mf.operator): mf.value
                    }
                }
                for mf in filters.filters
            ]
        }
    elif filters.condition == FilterCondition.OR:
        mql = {
            "$or": [
                {
                    prepare_key(mf.key): {
                        map_lc_mql_filter_operators(mf.operator): mf.value
                    }
                }
                for mf in filters.filters
            ]
        }
    else:
        logger.debug(f"filters.condition not recognized. Returning empty dict")
        mql = {}

    return mql


def vector_search_stage(
    query_vector: List[float],
    search_field: str,
    index_name: str,
    limit: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    oversampling_factor: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Vector Search Stage without Scores.

    Scoring is applied later depending on strategy.
    vector search includes a vectorSearchScore that is typically used.
    hybrid uses Reciprocal Rank Fusion.

    Args:
        query_vector: List of embedding vector
        search_field: Field in Collection containing embedding vectors
        index_name: Name of Atlas Vector Search Index tied to Collection
        limit: Number of documents to return
        oversampling_factor: this times limit is the number of candidates
        filters: Any MQL match expression comparing an indexed field

    Returns:
        Dictionary defining the $vectorSearch

    """
    if filter is None:
        filter = {}
    return {
        "$vectorSearch": {
            "index": index_name,
            "path": search_field,
            "queryVector": query_vector,
            "numCandidates": limit * oversampling_factor,
            "limit": limit,
            "filter": filter,
        }
    }


def combine_pipelines(
    pipeline: List[Any], stage: List[Dict[str, Any]], collection_name: str
):
    """Combines two aggregations into a single result set."""
    if pipeline:
        pipeline.append({"$unionWith": {"coll": collection_name, "pipeline": stage}})
    else:
        pipeline.extend(stage)
    return pipeline


def map_lc_mql_filter_operators(operator: FilterOperator) -> str:
    """Maps Llama-index FilterOperators to MongoDB Query Language."""
    operator_map = {
        FilterOperator.EQ: "$eq",  # = "=="  # default operator (string, int, float)
        FilterOperator.GT: "$gt",  # ">"  greater than (int, float)
        FilterOperator.LT: "$lt",  # = # "<"  # less than (int, float)
        FilterOperator.NE: "$ne",  # "!="  # not equal to (string, int, float)
        FilterOperator.GTE: "$gte",  # = ">="  # greater than or equal to (int, float)
        FilterOperator.LTE: "$lte",  # = "<="  # less than or equal to (int, float)
        FilterOperator.IN: "$in",  # = "in"  # metadata in value array (string or number)
        FilterOperator.NIN: "$nin",  # = "nin"  metadata not in value array (string, number)
        # FilterOperator.TEXT_MATCH: "NA", #  not supported as filter. See $text
        # FilterOperator.CONTAINS: "NA", # not supported as filter. Try $in
    }
    if operator not in operator_map:
        error_msg = f"Unsupported filter operator: {operator}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return operator_map[operator]


def reciprocal_rank_stage(score_field: str, penalty: float = 0) -> List[Dict[str, Any]]:
    r"""
    Stage adds Reciprocal Rank Fusion weighting.

        First, it pushes documents retrieved from previous stage
        into a temporary sub-document. It then unwinds to establish
        the rank to each and applies the penalty.

    Args:
        score_field: A unique string to identify the search being ranked.
        penalty: One method to weight scores in RRF.

    Returns:
        RRF score := \frac{1}{rank + penalty} with rank in [1,2,..,n]

    """
    return [
        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$addFields": {
                f"docs.{score_field}": {
                    "$divide": [1.0, {"$add": ["$rank", penalty, 1]}]
                },
                "docs.rank": "$rank",
                "_id": "$docs._id",
            }
        },
        {"$replaceRoot": {"newRoot": "$docs"}},
    ]


def final_hybrid_stage(
    scores_fields: List[str], limit: int, alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Sum weighted scores, sort, and apply limit.

    Args:
        scores_fields: List of fields given to scores of vector and text searches
        limit: Number of documents to return

    Returns:
        Final aggregation stages

    """
    assert set(scores_fields) == {"vector_score", "fulltext_score"}

    return [
        {"$group": {"_id": "$_id", "docs": {"$mergeObjects": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$docs"}},
        {"$set": {score: {"$ifNull": [f"${score}", 0]} for score in scores_fields}},
        {
            "$addFields": {
                "score": {
                    "$add": [
                        {"$multiply": [alpha, "$vector_score"]},
                        {"$multiply": [{"$subtract": [1.0, alpha]}, "$fulltext_score"]},
                    ]
                }
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": limit},
    ]
