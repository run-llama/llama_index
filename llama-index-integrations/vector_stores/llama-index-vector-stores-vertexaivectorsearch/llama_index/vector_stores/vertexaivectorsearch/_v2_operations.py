"""
V2 operations for Vertex AI Vector Search.

This module contains all v2-specific operations and is imported lazily
only when api_version="v2" is used.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStoreQueryMode,
    MetadataFilters,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict

_logger = logging.getLogger(__name__)

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


def _import_v2_sdk():
    """
    Import v2 SDK with proper error handling.

    Returns:
        The vectorsearch_v1beta module

    Raises:
        ImportError: If google-cloud-vectorsearch is not installed

    """
    try:
        from google.cloud import vectorsearch_v1beta

        return vectorsearch_v1beta
    except ImportError as e:
        raise ImportError(
            "v2 operations require google-cloud-vectorsearch. "
            "Install with: pip install google-cloud-vectorsearch"
        ) from e


def add_v2(
    store: Any,
    nodes: List[BaseNode],
    is_complete_overwrite: bool = False,
    **add_kwargs: Any,
) -> List[str]:
    """
    Add nodes to collection using v2 API.

    Args:
        store: The VertexAIVectorStore instance
        nodes: List of nodes with embeddings
        is_complete_overwrite: Whether to overwrite existing data
        **add_kwargs: Additional keyword arguments

    Returns:
        List of node IDs

    """
    vectorsearch = _import_v2_sdk()

    _logger.info(f"Adding {len(nodes)} nodes to v2 collection: {store.collection_id}")

    # Get or create v2 client
    from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=store.project_id,
        region=store.region,
        credentials_path=store.credentials_path,
    )

    # Get v2 clients
    clients = sdk_manager.get_v2_client()
    data_object_client = clients["data_object_service_client"]

    # Build parent path
    parent = f"projects/{store.project_id}/locations/{store.region}/collections/{store.collection_id}"

    # Convert nodes to v2 data objects
    batch_requests = []
    ids = []

    for node in nodes:
        node_id = node.node_id
        metadata = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)
        embedding = node.get_embedding()

        # Prepare data and vectors following the notebook pattern
        data_object = {
            "data": metadata,  # Metadata becomes the data field
            "vectors": {
                # Assuming default embedding field name
                "embedding": {"dense": {"values": embedding}}
            },
        }

        # Create batch request item
        batch_requests.append({"data_object_id": node_id, "data_object": data_object})
        ids.append(node_id)

    # Batch create data objects
    batch_size = store.batch_size
    for i in range(0, len(batch_requests), batch_size):
        batch = batch_requests[i : i + batch_size]
        _logger.info(f"Creating batch {i // batch_size + 1} ({len(batch)} objects)")

        request = vectorsearch.BatchCreateDataObjectsRequest(
            parent=parent, requests=batch
        )

        try:
            response = data_object_client.batch_create_data_objects(request)
            _logger.debug(f"Batch create response: {response}")
        except Exception as e:
            _logger.error(f"Failed to create data objects batch: {e}")
            raise

    return ids


def delete_v2(
    store: Any,
    ref_doc_id: str,
    **delete_kwargs: Any,
) -> None:
    """
    Delete nodes using ref_doc_id with v2 API.

    Args:
        store: The VertexAIVectorStore instance
        ref_doc_id: The document ID to delete
        **delete_kwargs: Additional keyword arguments

    """
    vectorsearch = _import_v2_sdk()

    _logger.info(f"Deleting nodes with ref_doc_id: {ref_doc_id} from v2 collection")

    # Get v2 client
    from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=store.project_id,
        region=store.region,
        credentials_path=store.credentials_path,
    )
    clients = sdk_manager.get_v2_client()
    data_object_client = clients["data_object_service_client"]
    search_client = clients["data_object_search_service_client"]

    # Build parent path
    parent = f"projects/{store.project_id}/locations/{store.region}/collections/{store.collection_id}"

    # Query for data objects with matching ref_doc_id
    query_request = vectorsearch.QueryDataObjectsRequest(
        parent=parent,
        filter={"ref_doc_id": {"$eq": ref_doc_id}},
        output_fields=vectorsearch.OutputFields(
            data_fields=["ref_doc_id"], metadata_fields=["*"]
        ),
    )

    try:
        # Execute query
        results = search_client.query_data_objects(query_request)

        # Build batch delete requests
        delete_requests = []
        for data_object in results:
            delete_requests.append(
                vectorsearch.DeleteDataObjectRequest(name=data_object.name)
            )

        # Batch delete
        if delete_requests:
            batch_delete_request = vectorsearch.BatchDeleteDataObjectsRequest(
                parent=parent, requests=delete_requests
            )
            response = data_object_client.batch_delete_data_objects(
                batch_delete_request
            )
            _logger.info(
                f"Deleted {len(delete_requests)} data objects with ref_doc_id: {ref_doc_id}"
            )
        else:
            _logger.info(f"No data objects found with ref_doc_id: {ref_doc_id}")
    except Exception as e:
        _logger.error(f"Failed to delete data objects: {e}")
        raise


def query_v2(
    store: Any,
    query: VectorStoreQuery,
    **kwargs: Any,
) -> VectorStoreQueryResult:
    """
    Query collection for top k most similar nodes using v2 API.

    Args:
        store: The VertexAIVectorStore instance
        query: The vector store query
        **kwargs: Additional keyword arguments

    Returns:
        Vector store query result

    """
    vectorsearch = _import_v2_sdk()

    _logger.info(f"Querying v2 collection: {store.collection_id}")

    # Get v2 client
    from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=store.project_id,
        region=store.region,
        credentials_path=store.credentials_path,
    )
    clients = sdk_manager.get_v2_client()
    search_client = clients["data_object_search_service_client"]

    # Build parent path
    parent = f"projects/{store.project_id}/locations/{store.region}/collections/{store.collection_id}"

    # Extract query embedding
    query_embedding = None
    if query.mode == VectorStoreQueryMode.DEFAULT:
        query_embedding = query.query_embedding

    if query_embedding is None:
        _logger.warning("No query embedding provided, returning empty result")
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

    # Build search request following notebook pattern
    search_request = vectorsearch.SearchDataObjectsRequest(
        parent=parent,
        vector_search=vectorsearch.VectorSearch(
            search_field="embedding",  # Default field name
            vector=vectorsearch.DenseVector(values=query_embedding),
            top_k=query.similarity_top_k,
            output_fields=vectorsearch.OutputFields(
                data_fields=["*"], vector_fields=["*"], metadata_fields=["*"]
            ),
        ),
    )

    # Add filters if provided
    if query.filters is not None:
        # TODO: Convert LlamaIndex filters to v2 filter format
        pass

    try:
        # Execute search
        results = search_client.search_data_objects(search_request)

        # Process response and convert to TextNodes
        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for result in results:
            data_obj = result.data_object

            # Extract embedding vector if available
            embedding = None
            if hasattr(data_obj, "vectors") and data_obj.vectors:
                # Access the embedding field from the vectors map
                if "embedding" in data_obj.vectors:
                    vector_data = data_obj.vectors["embedding"]
                    if hasattr(vector_data, "dense") and vector_data.dense:
                        if hasattr(vector_data.dense, "values"):
                            embedding = list(vector_data.dense.values)

            # Convert data to dict if it's a protobuf message
            metadata = dict(data_obj.data) if data_obj.data else {}

            node = TextNode(
                text=metadata.get(store.text_key, ""),
                id_=data_obj.name.split("/")[-1],  # Extract ID from resource name
                metadata=metadata,
                embedding=embedding,
            )
            top_k_nodes.append(node)
            top_k_ids.append(data_obj.name.split("/")[-1])
            top_k_scores.append(result.score if hasattr(result, "score") else 1.0)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
    except Exception as e:
        _logger.error(f"Failed to search data objects: {e}")
        raise


def delete_nodes_v2(
    store: Any,
    node_ids: Optional[List[str]] = None,
    filters: Optional[MetadataFilters] = None,
    **kwargs: Any,
) -> None:
    """
    Delete nodes by IDs or filters using v2 API.

    Args:
        store: The VertexAIVectorStore instance
        node_ids: List of node IDs to delete
        filters: Metadata filters to select nodes for deletion
        **kwargs: Additional keyword arguments

    """
    vectorsearch = _import_v2_sdk()

    _logger.info(f"Deleting nodes from v2 collection: {store.collection_id}")

    # Get v2 client
    from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=store.project_id,
        region=store.region,
        credentials_path=store.credentials_path,
    )
    clients = sdk_manager.get_v2_client()
    data_object_client = clients["data_object_service_client"]
    search_client = clients["data_object_search_service_client"]

    # Build parent path
    parent = f"projects/{store.project_id}/locations/{store.region}/collections/{store.collection_id}"
    collection_name = parent

    if node_ids is not None:
        # Delete by node IDs
        _logger.info(f"Deleting {len(node_ids)} nodes by ID")

        # Build batch delete requests
        delete_requests = []
        for node_id in node_ids:
            delete_requests.append(
                vectorsearch.DeleteDataObjectRequest(
                    name=f"{collection_name}/dataObjects/{node_id}"
                )
            )

        try:
            if delete_requests:
                batch_delete_request = vectorsearch.BatchDeleteDataObjectsRequest(
                    parent=parent, requests=delete_requests
                )
                response = data_object_client.batch_delete_data_objects(
                    batch_delete_request
                )
                _logger.info(f"Deleted {len(delete_requests)} data objects by ID")
            else:
                _logger.info("No data objects to delete")
        except Exception as e:
            _logger.error(f"Failed to delete data objects by ID: {e}")
            raise
    elif filters is not None:
        # Delete by filters - need to query first then delete
        _logger.info(f"Deleting nodes matching filters")

        # For now, we'll skip filter conversion and just log
        _logger.warning(
            "Filter-based deletion not yet implemented. "
            "LlamaIndex MetadataFilters need conversion to v2 filter format."
        )
        # TODO: Implement filter conversion when we understand the mapping better
        # This would require converting LlamaIndex MetadataFilters to v2's filter format
    else:
        raise ValueError("Either node_ids or filters must be provided")


def clear_v2(store: Any) -> None:
    """
    Clear all nodes from the collection using v2 API.

    Args:
        store: The VertexAIVectorStore instance

    """
    vectorsearch = _import_v2_sdk()

    _logger.info(f"Clearing all nodes from v2 collection: {store.collection_id}")

    # Get v2 client
    from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=store.project_id,
        region=store.region,
        credentials_path=store.credentials_path,
    )
    clients = sdk_manager.get_v2_client()
    data_object_client = clients["data_object_service_client"]
    search_client = clients["data_object_search_service_client"]

    # Build parent path
    parent = f"projects/{store.project_id}/locations/{store.region}/collections/{store.collection_id}"
    collection_name = parent

    try:
        # Query all data objects (without filter to get all)
        query_request = vectorsearch.QueryDataObjectsRequest(
            parent=parent,
            page_size=100,  # Process in batches
            output_fields=vectorsearch.OutputFields(metadata_fields=["*"]),
        )

        # Iterate through pages and delete all
        total_deleted = 0
        paged_response = search_client.query_data_objects(query_request)
        for page in paged_response.pages:
            delete_requests = []
            for data_object in page.data_objects:
                delete_requests.append(
                    vectorsearch.DeleteDataObjectRequest(name=data_object.name)
                )

            # Batch delete this page
            if delete_requests:
                batch_delete_request = vectorsearch.BatchDeleteDataObjectsRequest(
                    parent=parent, requests=delete_requests
                )
                data_object_client.batch_delete_data_objects(batch_delete_request)
                total_deleted += len(delete_requests)

        _logger.info(f"Cleared {total_deleted} data objects from collection")
    except Exception as e:
        _logger.error(f"Failed to clear collection: {e}")
        raise
