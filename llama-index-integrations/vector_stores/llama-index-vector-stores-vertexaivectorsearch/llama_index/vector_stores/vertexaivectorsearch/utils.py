"""
Vertex AI Vector Search specific serializers for LlamaIndex data structures.

Contain conversion to and from dataclasses that LlamaIndex uses.

"""

import json
import logging
import time
import uuid
import warnings
from functools import wraps
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from google.api_core.gapic_v1.client_info import ClientInfo
from google.cloud.aiplatform.compat.types import (  # type: ignore[attr-defined, unused-ignore]
    matching_engine_index as meidx_types,
)
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchNeighbor,
    Namespace,
    NumericNamespace,
)
from google.cloud.storage import Bucket  # type: ignore[import-untyped, unused-ignore]
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
)

_logger = logging.getLogger(__name__)

FILTER_MAP = {
    FilterOperator.EQ: "EQUAL",
    FilterOperator.LT: "LESS",
    FilterOperator.LTE: "LESS_EQUAL",
    FilterOperator.GT: "GREATER",
    FilterOperator.GTE: "GREATER_EQUAL",
    FilterOperator.NE: "NOT_EQUAL",
}

MAX_DATA_POINTS = 10000


def _import_vertexai(minimum_expected_version: str = "1.44.0") -> Any:
    """
    Try to import pinecone module. If it's not already installed, instruct user how to install.
    """
    try:
        from google.cloud import aiplatform
    except ImportError as e:
        raise ImportError(
            "Please, install or upgrade the google-cloud-aiplatform library: "
            f"pip install google-cloud-aiplatform>={minimum_expected_version}"
        ) from e
    return aiplatform


def get_user_agent(module: Optional[str] = None) -> Tuple[str, str]:
    r"""
    Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.

    Returns:
        Tuple[str, str]

    """
    try:
        llamaindex_version = metadata.version(
            "llama-index-vector-stores-vertexaivectorsearch"
        )
    except metadata.PackageNotFoundError:
        llamaindex_version = "0.0.0"
    client_library_version = (
        f"{llamaindex_version}-{module}" if module else llamaindex_version
    )
    return (
        client_library_version,
        f"llama-index-vector-stores-vertexaivectorsearch/{client_library_version}",
    )


def get_client_info(module: Optional[str] = None) -> "ClientInfo":
    r"""
    Returns a client info object with a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.

    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo

    """
    client_library_version, user_agent = get_user_agent(module)
    return ClientInfo(  # type: ignore[no-untyped-call]
        client_library_version=client_library_version,
        user_agent=user_agent,
    )


def to_node(match: MatchNeighbor, text_key: str) -> TextNode:
    """Convert to Node."""
    entry = {}
    node_content = {}
    if match.restricts:
        entry = {
            namespace.name: namespace.allow_tokens for namespace in match.restricts
        }
        if "_node_content" in entry:
            entry["_node_content"] = entry["_node_content"][0]
            node_content = json.loads(entry["_node_content"])

    id = match.id
    embedding = list(match.feature_vector)
    text = node_content.get(text_key, "")

    try:
        node = metadata_dict_to_node(entry)
        node.text = text
        node.embedding = embedding
    except Exception as e:
        _logger.debug("Failed to parse Node metadata, fallback to legacy logic. %s", e)
        metadata, node_info, relationships = legacy_metadata_dict_to_node(entry)

        node = TextNode(
            text=text,
            id_=id,
            metadata=metadata,
            start_char_idx=node_info.get("start", None),
            end_char_idx=node_info.get("end", None),
            relationships=relationships,
            embedding=embedding,
        )
    return node


def stream_update_index(
    index: MatchingEngineIndex, data_points: List["meidx_types.IndexDataPoint"]
) -> None:
    """
    Updates an index using stream updating.

    Args:
        index: Vector search index.
        data_points: List of IndexDataPoint.

    """
    index.upsert_datapoints(data_points)


def batch_update_index(
    index: MatchingEngineIndex,
    data_points: List["meidx_types.IndexDataPoint"],
    *,
    staging_bucket: Bucket,
    prefix: Union[str, None] = None,
    file_name: str = "documents.json",
    is_complete_overwrite: bool = False,
) -> None:
    """
    Updates an index using batch updating.

    Args:
        index: Vector search index.
        data_points: List of IndexDataPoint.
        staging_bucket: Bucket where the staging data is stored. Must be in the same
            region as the index.
        prefix: Prefix for the blob name. If not provided an unique iid will be
            generated.
        file_name: File name of the staging embeddings. By default 'documents.json'.
        is_complete_overwrite: Whether is an append or overwrite operation.

    """
    if prefix is None:
        prefix = str(uuid.uuid4())

    records = data_points_to_batch_update_records(data_points)

    file_content = "\n".join(json.dumps(record) for record in records)

    blob = staging_bucket.blob(f"index/{prefix}/{file_name}")
    blob.upload_from_string(file_content)

    contents_delta_uri = f"gs://{staging_bucket.name}/index/{prefix}"

    index.update_embeddings(
        contents_delta_uri=contents_delta_uri,
        is_complete_overwrite=is_complete_overwrite,
    )


def to_data_points(
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: Union[List[Dict[str, Any]], None],
) -> List["meidx_types.IndexDataPoint"]:
    """
    Converts triplets id, embedding, metadata into IndexDataPoints instances.

    Only metadata with values of type string, numeric or list of string will be
    considered for the filtering.

    Args:
        ids: List of unique ids.
        embeddings: List of feature representatitons.
        metadatas: List of metadatas.

    """
    if metadatas is None:
        metadatas = [{}] * len(ids)

    data_points = []
    ignored_fields = set()

    for id_, embedding, metadata in zip(ids, embeddings, metadatas):
        restricts = []
        numeric_restricts = []

        for namespace, value in metadata.items():
            if not isinstance(namespace, str):
                raise ValueError("All metadata keys must be strings")

            if isinstance(value, str):
                restriction = meidx_types.IndexDatapoint.Restriction(
                    namespace=namespace, allow_list=[value]
                )
                restricts.append(restriction)
            elif isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                restriction = meidx_types.IndexDatapoint.Restriction(
                    namespace=namespace, allow_list=value
                )
                restricts.append(restriction)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                restriction = meidx_types.IndexDatapoint.NumericRestriction(
                    namespace=namespace, value_float=value
                )
                numeric_restricts.append(restriction)
            else:
                ignored_fields.add(namespace)

        if len(ignored_fields) > 0:
            warnings.warn(
                f"Some values in fields {', '.join(ignored_fields)} are not usable for"
                f" restrictions. In order to be used they must be str, list[str] or"
                f" numeric."
            )

        data_point = meidx_types.IndexDatapoint(
            datapoint_id=id_,
            feature_vector=embedding,
            restricts=restricts,
            numeric_restricts=numeric_restricts,
        )

        data_points.append(data_point)

    return data_points


def data_points_to_batch_update_records(
    data_points: List["meidx_types.IndexDataPoint"],
) -> List[Dict[str, Any]]:
    """
    Given a list of datapoints, generates a list of records in the input format
    required to do a bactch update.

    Args:
        data_points: List of IndexDataPoints.

    Returns:
        List of records with the format needed to do a batch update.

    """
    records = []

    for data_point in data_points:
        record = {
            "id": data_point.datapoint_id,
            "embedding": list(data_point.feature_vector),
            "restricts": [
                {
                    "namespace": restrict.namespace,
                    "allow": list(restrict.allow_list),
                }
                for restrict in data_point.restricts
            ],
            "numeric_restricts": [
                {"namespace": restrict.namespace, "value_float": restrict.value_float}
                for restrict in data_point.numeric_restricts
            ],
        }

        records.append(record)

    return records


def find_neighbors(
    index: MatchingEngineIndex,
    endpoint: MatchingEngineIndexEndpoint,
    embeddings: List[List[float]],
    top_k: int = 4,
    filter: Union[List[Namespace], None] = None,
    numeric_filter: Union[List[NumericNamespace], None] = None,
    return_full_datapoint: bool = True,
) -> List[MatchNeighbor]:
    """
    Finds the k closes neighbors of each instance of embeddings.

    Args:
        embedding: List of embeddings vectors.
        k: Number of neighbors to be retrieved.
        filter_: List of filters to apply.

    Returns:
        List of lists of Tuples (id, distance) for each embedding vector.

    """
    # No need to implement other method for private VPC, find_neighbors now works
    # with public and private.
    neighbors = endpoint.find_neighbors(
        deployed_index_id=_get_deployed_index_id(index, endpoint),
        queries=embeddings,
        num_neighbors=top_k,
        filter=filter,
        numeric_filter=numeric_filter,
        return_full_datapoint=True,
    )

    if len(neighbors) > 0:
        neighbors = neighbors[0]

    return neighbors


def _get_deployed_index_id(
    index: MatchingEngineIndex, endpoint: MatchingEngineIndexEndpoint
) -> str:
    """
    Gets the deployed index id that matches with the provided index.

    Raises:
        ValueError if the index provided is not found in the endpoint.

    """
    for deployed_index in endpoint.deployed_indexes:
        if deployed_index.index == index.resource_name:
            return deployed_index.id

    raise ValueError(
        f"No index with id {index.resource_name} "
        f"deployed on endpoint "
        f"{endpoint.display_name}."
    )


def to_vectorsearch_filter(filters: MetadataFilters):  # type: ignore
    """
    Converts llamaindex filters to Vertex AI Vector Search filter syntax
    based on data type of value and operator passed in filters.

    Raises:
        ValueError when invalid operator is passed to the filter.

    """
    if filters:
        num_filters = []
        txt_filters = []
        for filter in filters.filters:
            num_filter = None
            txt_filter = None
            if filter.operator not in FILTER_MAP:
                raise ValueError(
                    "Invalid operator for filters. "
                    f"Supported operators are: {FILTER_MAP.keys()}"
                )
            op = FILTER_MAP[filter.operator]
            if isinstance(filter.value, int):
                num_filter = NumericNamespace(
                    name=filter.key, value_int=filter.value, op=op
                )
            elif isinstance(filter.value, float):
                num_filter = NumericNamespace(
                    name=filter.key, value_float=filter.value, op=op
                )
            else:
                txt_filter = Namespace(name=filter.key, allow_tokens=[filter.value])

            # return filters
            if txt_filter:
                txt_filters.append(txt_filter)
            if num_filter:
                num_filters.append(num_filter)

        return txt_filters, num_filters

    else:
        return None


def get_datapoints_by_filter(
    index: MatchingEngineIndex,
    endpoint: MatchingEngineIndexEndpoint,
    metadata: dict = {},
    max_datapoints: int = MAX_DATA_POINTS,
) -> List[str]:
    """
    Gets all the datapoints matching the metadata filters (text only)
    on the specified deployed index.
    """
    # configure filter based on metadata
    index_config = index.to_dict()["metadata"]["config"]
    embeddings = [[0.0] * int(index_config.get("dimensions", 1))]
    filter = None
    if metadata:
        filter = [
            Namespace(name=key, allow_tokens=[value]) for key, value in metadata.items()
        ]

    # Find datapoints matching the filter expression and return datapoint ids
    try:
        neighbors = endpoint.find_neighbors(
            deployed_index_id=_get_deployed_index_id(index, endpoint),
            queries=embeddings,
            num_neighbors=max_datapoints,
            filter=filter,
            return_full_datapoint=False,
        )

        data_points = [neighbor.id for neighbor in neighbors[0]]
    except Exception as e:
        _logger.error("Failed to query datapoint due to error: %s", e)
        data_points = []

    return data_points


T = TypeVar("T")


def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """
    Simple retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay in seconds (doubles on each retry)

    Returns:
        Decorator function

    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        # Last attempt, raise the exception
                        raise
                    # Exponential backoff
                    sleep_time = delay * (2**attempt)
                    _logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
            # This should never be reached, but satisfies type checker
            raise RuntimeError("Retry logic error")

        return wrapper

    return decorator


# =============================================================================
# Helper Functions for Hybrid Search
# =============================================================================


def _calculate_rrf_weights(alpha: float, num_searches: int = 2) -> List[float]:
    """
    Calculate RRF weights from alpha value.

    Args:
        alpha: Weight for vector search (0 = text only, 1 = vector only)
        num_searches: Number of searches being combined

    Returns:
        List of weights [vector_weight, text_weight]

    """
    if num_searches == 2:
        return [alpha, 1.0 - alpha]
    return [1.0 / num_searches] * num_searches


def _convert_filters_to_v2(filters: Optional[MetadataFilters]) -> Optional[dict]:
    """
    Convert LlamaIndex MetadataFilters to V2 filter format.

    V2 filter format:
    - Simple: {"field": {"$eq": "value"}}
    - AND: {"$and": [{...}, {...}]}
    - OR: {"$or": [{...}, {...}]}

    Args:
        filters: LlamaIndex MetadataFilters object

    Returns:
        V2 filter dict or None

    """
    if filters is None or len(filters.filters) == 0:
        return None

    op_map = {
        FilterOperator.EQ: "$eq",
        FilterOperator.NE: "$ne",
        FilterOperator.GT: "$gt",
        FilterOperator.GTE: "$gte",
        FilterOperator.LT: "$lt",
        FilterOperator.LTE: "$lte",
        FilterOperator.IN: "$in",
        FilterOperator.NIN: "$nin",
        FilterOperator.CONTAINS: "$contains",
    }

    def convert_single(f: MetadataFilter) -> dict:
        v2_op = op_map.get(f.operator, "$eq")
        return {f.key: {v2_op: f.value}}

    if len(filters.filters) == 1:
        f = filters.filters[0]
        if isinstance(f, MetadataFilters):
            return _convert_filters_to_v2(f)
        return convert_single(f)

    converted = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            converted.append(_convert_filters_to_v2(f))
        else:
            converted.append(convert_single(f))

    condition = "$and" if filters.condition == FilterCondition.AND else "$or"
    return {condition: converted}
