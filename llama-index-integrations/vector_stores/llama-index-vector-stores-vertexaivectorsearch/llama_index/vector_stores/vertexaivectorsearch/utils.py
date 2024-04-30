"""Vertex AI Vector Search specific serializers for LlamaIndex data structures.

Contain conversion to and from dataclasses that LlamaIndex uses.

"""

import json
import logging
import warnings
from importlib import metadata
from typing import Any, Dict, List, Optional, Union, Tuple

import uuid


from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
)

from llama_index.core.vector_stores.types import MetadataFilters

from google.api_core.gapic_v1.client_info import ClientInfo

from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchNeighbor,
    Namespace,
    NumericNamespace,
)

from google.cloud import aiplatform_v1 as aip

from google.cloud.aiplatform.compat.types import (  # type: ignore[attr-defined, unused-ignore]
    matching_engine_index as meidx_types,
)
from google.cloud.storage import Bucket  # type: ignore[import-untyped, unused-ignore]

_logger = logging.getLogger(__name__)


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
    r"""Returns a custom user agent header.

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
    r"""Returns a client info object with a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.

    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo
    """
    client_library_version, user_agent = get_user_agent(module)
    return ClientInfo(
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
    """Updates an index using stream updating.

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
    """Updates an index using batch updating.

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

    blob = staging_bucket.blob(f"{prefix}/{file_name}")
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
    """Converts triplets id, embedding, metadata into IndexDataPoints instances.

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
    """Given a list of datapoints, generates a list of records in the input format
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
    filter_: Union[List[Namespace], None] = None,
    numeric_filter: Union[List[NumericNamespace], None] = None,
) -> List[MatchNeighbor]:
    """Finds the k closes neighbors of each instance of embeddings.

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
        filter=filter_,
        numeric_filter=numeric_filter,
        return_full_datapoint=True,
    )

    if len(neighbors) > 0:
        neighbors = neighbors[0]

    return neighbors


def _get_deployed_index_id(
    index: MatchingEngineIndex, endpoint: MatchingEngineIndexEndpoint
) -> str:
    """Gets the deployed index id that matches with the provided index.

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


def to_vectorsearch_filter(filters: MetadataFilters) -> List[Namespace]:
    if filters:
        return [
            Namespace(name=filter.key, allow_tokens=[filter.value])
            for filter in filters.filters
        ]
    else:
        return None


def get_datapoint(
    index: MatchingEngineIndex, endpoint: MatchingEngineIndexEndpoint, datapoint_id: str
):
    # Set variables for the current deployed index.
    if endpoint.private_service_access_network:
        api_endpoint = endpoint.private_service_access_network
    elif endpoint.public_endpoint_domain_name:
        api_endpoint = endpoint.public_endpoint_domain_name
    else:
        api_endpoint = None

    # Configure Vector Search client
    client_options = {"api_endpoint": api_endpoint}
    vector_search_client = aip.MatchServiceClient(client_options=client_options)

    # Build FindNeighborsRequest object
    datapoint = aip.IndexDatapoint(datapoint_id=datapoint_id)
    query = aip.FindNeighborsRequest.Query(datapoint=datapoint, neighbor_count=1)
    request = aip.FindNeighborsRequest(
        index_endpoint=endpoint.resource_name,
        deployed_index_id=_get_deployed_index_id(index, endpoint),
        queries=[query],
        return_full_datapoint=False,
    )

    # Execute the request and return the result
    try:
        result = vector_search_client.find_neighbors(request)
    except Exception as e:
        _logger.error("Failed to query datapoint due to error: %s", e)
        result = []

    return result
