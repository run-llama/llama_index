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

    from llama_index.core.vector_stores.types import (
        FilterOperator,
        FilterCondition,
        MetadataFilter,
    )

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


def _merge_results_rrf(
    store: Any,
    vector_results: List[Any],
    text_results: List[Any],
    alpha: float,
    top_k: int,
    k: int = 60,
) -> VectorStoreQueryResult:
    """
    Merge vector and text search results using Reciprocal Rank Fusion.

    RRF formula: score = sum(1 / (k + rank_i)) for each result list
    With alpha weighting: final_score = alpha * vector_rrf + (1-alpha) * text_rrf

    Args:
        store: VertexAIVectorStore instance
        vector_results: Results from vector search
        text_results: Results from text search
        alpha: Weight for vector results (0=text only, 1=vector only)
        top_k: Number of results to return
        k: RRF constant (default 60)

    Returns:
        Merged VectorStoreQueryResult

    """
    # Build score maps: id -> (rrf_score, result_object)
    vector_scores = {}
    for rank, result in enumerate(vector_results):
        node_id = result.data_object.name.split("/")[-1]
        rrf_score = 1.0 / (k + rank + 1)
        vector_scores[node_id] = (rrf_score * alpha, result)

    text_scores = {}
    for rank, result in enumerate(text_results):
        node_id = result.data_object.name.split("/")[-1]
        rrf_score = 1.0 / (k + rank + 1)
        text_scores[node_id] = (rrf_score * (1 - alpha), result)

    # Merge scores
    all_ids = set(vector_scores.keys()) | set(text_scores.keys())
    merged_scores = []
    for node_id in all_ids:
        v_score, v_result = vector_scores.get(node_id, (0, None))
        t_score, t_result = text_scores.get(node_id, (0, None))
        total_score = v_score + t_score
        result = v_result or t_result
        merged_scores.append((total_score, node_id, result))

    # Sort by score descending and take top_k
    merged_scores.sort(key=lambda x: x[0], reverse=True)
    merged_scores = merged_scores[:top_k]

    # Build result
    nodes = []
    ids = []
    similarities = []

    for score, node_id, result in merged_scores:
        data_obj = result.data_object

        # Extract embedding
        embedding = None
        if hasattr(data_obj, "vectors") and data_obj.vectors:
            if store.embedding_field in data_obj.vectors:
                vector_data = data_obj.vectors[store.embedding_field]
                if hasattr(vector_data, "dense") and vector_data.dense:
                    if hasattr(vector_data.dense, "values"):
                        embedding = list(vector_data.dense.values)

        # Extract metadata
        metadata = dict(data_obj.data) if data_obj.data else {}

        node = TextNode(
            text=metadata.get(store.text_key, ""),
            id_=node_id,
            metadata=metadata,
            embedding=embedding,
        )

        nodes.append(node)
        ids.append(node_id)
        similarities.append(score)

    return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)


def _build_ranker(
    store: Any,
    query: VectorStoreQuery,
    num_searches: int,
) -> Any:
    """
    Build the appropriate ranker based on store configuration.

    Args:
        store: VertexAIVectorStore instance
        query: The query being executed
        num_searches: Number of searches being combined

    Returns:
        Ranker object (RRF or VertexRanker)

    """
    vectorsearch = _import_v2_sdk()

    if store.hybrid_ranker == "vertex":
        title_template = None
        content_template = None
        if store.vertex_ranker_title_field:
            title_template = f"{{{{ {store.vertex_ranker_title_field} }}}}"
        if store.vertex_ranker_content_field:
            content_template = f"{{{{ {store.vertex_ranker_content_field} }}}}"

        return vectorsearch.Ranker(
            vertex=vectorsearch.VertexRanker(
                query=query.query_str or "",
                model=store.vertex_ranker_model,
                title_template=title_template,
                content_template=content_template,
            )
        )
    else:
        # RRF ranker (default)
        alpha = query.alpha if query.alpha is not None else store.default_hybrid_alpha
        weights = _calculate_rrf_weights(alpha, num_searches)
        return vectorsearch.Ranker(
            rrf=vectorsearch.ReciprocalRankFusion(weights=weights)
        )


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


# =============================================================================
# Result Processing Helpers
# =============================================================================


def _process_search_results(
    store: Any,
    results: Any,
) -> VectorStoreQueryResult:
    """
    Process single search results into VectorStoreQueryResult.

    Args:
        store: VertexAIVectorStore instance
        results: Search results from the API

    Returns:
        VectorStoreQueryResult with nodes, similarities, and IDs

    """
    top_k_nodes = []
    top_k_ids = []
    top_k_scores = []

    for result in results:
        data_obj = result.data_object

        # Extract embedding
        embedding = None
        if hasattr(data_obj, "vectors") and data_obj.vectors:
            if store.embedding_field in data_obj.vectors:
                vector_data = data_obj.vectors[store.embedding_field]
                if hasattr(vector_data, "dense") and vector_data.dense:
                    if hasattr(vector_data.dense, "values"):
                        embedding = list(vector_data.dense.values)

        # Extract metadata
        metadata = dict(data_obj.data) if data_obj.data else {}

        # Extract ID from resource name or data_object_id
        node_id = data_obj.name.split("/")[-1]
        if hasattr(data_obj, "data_object_id") and data_obj.data_object_id:
            node_id = data_obj.data_object_id

        node = TextNode(
            text=metadata.get(store.text_key, ""),
            id_=node_id,
            metadata=metadata,
            embedding=embedding,
        )

        top_k_nodes.append(node)
        top_k_ids.append(node_id)
        # Use distance or score depending on what's available
        score = 1.0
        if hasattr(result, "distance"):
            score = result.distance
        elif hasattr(result, "score"):
            score = result.score
        top_k_scores.append(score)

    return VectorStoreQueryResult(
        nodes=top_k_nodes,
        similarities=top_k_scores,
        ids=top_k_ids,
    )


def _process_batch_search_results(
    store: Any,
    batch_results: Any,
    top_k: int,
) -> VectorStoreQueryResult:
    """
    Process batch search (RRF/VertexRanker combined) results.

    Args:
        store: VertexAIVectorStore instance
        batch_results: Batch search results from the API
        top_k: Maximum number of results to return

    Returns:
        VectorStoreQueryResult with nodes, similarities, and IDs

    """
    top_k_nodes = []
    top_k_ids = []
    top_k_scores = []

    # Batch results contain combined ranked results
    # The structure may vary based on API response format
    if hasattr(batch_results, "combined_results") and batch_results.combined_results:
        # Process combined results from RRF/ranker
        for result in batch_results.combined_results:
            if len(top_k_nodes) >= top_k:
                break

            data_obj = result.data_object

            # Extract embedding
            embedding = None
            if hasattr(data_obj, "vectors") and data_obj.vectors:
                if store.embedding_field in data_obj.vectors:
                    vector_data = data_obj.vectors[store.embedding_field]
                    if hasattr(vector_data, "dense") and vector_data.dense:
                        if hasattr(vector_data.dense, "values"):
                            embedding = list(vector_data.dense.values)

            # Extract metadata
            metadata = dict(data_obj.data) if data_obj.data else {}

            # Extract ID
            node_id = data_obj.name.split("/")[-1]
            if hasattr(data_obj, "data_object_id") and data_obj.data_object_id:
                node_id = data_obj.data_object_id

            node = TextNode(
                text=metadata.get(store.text_key, ""),
                id_=node_id,
                metadata=metadata,
                embedding=embedding,
            )

            top_k_nodes.append(node)
            top_k_ids.append(node_id)
            score = 1.0
            if hasattr(result, "distance"):
                score = result.distance
            elif hasattr(result, "score"):
                score = result.score
            elif hasattr(result, "rank_score"):
                score = result.rank_score
            top_k_scores.append(score)
    elif hasattr(batch_results, "results"):
        # Alternative response structure
        for response in batch_results.results:
            if hasattr(response, "results"):
                for result in response.results:
                    if len(top_k_nodes) >= top_k:
                        break

                    data_obj = result.data_object

                    # Extract embedding
                    embedding = None
                    if hasattr(data_obj, "vectors") and data_obj.vectors:
                        if store.embedding_field in data_obj.vectors:
                            vector_data = data_obj.vectors[store.embedding_field]
                            if hasattr(vector_data, "dense") and vector_data.dense:
                                if hasattr(vector_data.dense, "values"):
                                    embedding = list(vector_data.dense.values)

                    # Extract metadata
                    metadata = dict(data_obj.data) if data_obj.data else {}

                    # Extract ID
                    node_id = data_obj.name.split("/")[-1]
                    if hasattr(data_obj, "data_object_id") and data_obj.data_object_id:
                        node_id = data_obj.data_object_id

                    node = TextNode(
                        text=metadata.get(store.text_key, ""),
                        id_=node_id,
                        metadata=metadata,
                        embedding=embedding,
                    )

                    top_k_nodes.append(node)
                    top_k_ids.append(node_id)
                    score = 1.0
                    if hasattr(result, "distance"):
                        score = result.distance
                    elif hasattr(result, "score"):
                        score = result.score
                    top_k_scores.append(score)

    return VectorStoreQueryResult(
        nodes=top_k_nodes,
        similarities=top_k_scores,
        ids=top_k_ids,
    )


# =============================================================================
# Query Mode Implementations
# =============================================================================


def _query_v2_default(
    store: Any,
    query: VectorStoreQuery,
    clients: dict,
    parent: str,
    **kwargs: Any,
) -> VectorStoreQueryResult:
    """
    Execute default dense vector search.

    Args:
        store: VertexAIVectorStore instance
        query: The vector store query
        clients: V2 client dictionary
        parent: Parent resource path
        **kwargs: Additional arguments

    Returns:
        VectorStoreQueryResult

    """
    vectorsearch = _import_v2_sdk()
    search_client = clients["data_object_search_service_client"]

    if query.query_embedding is None:
        raise ValueError(
            "query_embedding is required for DEFAULT (vector) search mode. "
            "Use TEXT_SEARCH mode if you only have a text query."
        )

    # Build filter
    v2_filter = _convert_filters_to_v2(query.filters)

    # Build search request
    search_kwargs = {
        "parent": parent,
        "vector_search": vectorsearch.VectorSearch(
            search_field=store.embedding_field,
            vector=vectorsearch.DenseVector(values=query.query_embedding),
            top_k=query.similarity_top_k,
            output_fields=vectorsearch.OutputFields(
                data_fields=["*"],
                vector_fields=["*"],
                metadata_fields=["*"],
            ),
        ),
    }

    if v2_filter:
        search_kwargs["vector_search"].filter = v2_filter

    search_request = vectorsearch.SearchDataObjectsRequest(**search_kwargs)

    try:
        results = search_client.search_data_objects(search_request)
        return _process_search_results(store, results)
    except Exception as e:
        _logger.error(f"Failed to execute DEFAULT search: {e}")
        raise


def _query_v2_text_search(
    store: Any,
    query: VectorStoreQuery,
    clients: dict,
    parent: str,
    **kwargs: Any,
) -> VectorStoreQueryResult:
    """
    Execute full-text keyword search only.

    Args:
        store: VertexAIVectorStore instance
        query: The vector store query
        clients: V2 client dictionary
        parent: Parent resource path
        **kwargs: Additional arguments

    Returns:
        VectorStoreQueryResult

    """
    vectorsearch = _import_v2_sdk()
    search_client = clients["data_object_search_service_client"]

    if query.query_str is None:
        raise ValueError("TEXT_SEARCH mode requires query_str.")

    if store.text_search_fields is None:
        raise ValueError(
            "TEXT_SEARCH mode requires text_search_fields to be configured "
            "in the constructor."
        )

    top_k = query.sparse_top_k or query.similarity_top_k

    # Build search request
    search_request = vectorsearch.SearchDataObjectsRequest(
        parent=parent,
        text_search=vectorsearch.TextSearch(
            search_text=query.query_str,
            data_field_names=store.text_search_fields,
            top_k=top_k,
            output_fields=vectorsearch.OutputFields(
                data_fields=["*"],
                vector_fields=["*"],
                metadata_fields=["*"],
            ),
        ),
    )

    try:
        results = search_client.search_data_objects(search_request)
        return _process_search_results(store, results)
    except Exception as e:
        _logger.error(f"Failed to execute TEXT_SEARCH: {e}")
        raise


def _query_v2_hybrid(
    store: Any,
    query: VectorStoreQuery,
    clients: dict,
    parent: str,
    **kwargs: Any,
) -> VectorStoreQueryResult:
    """
    Execute hybrid search: VectorSearch + TextSearch with ranker.

    Args:
        store: VertexAIVectorStore instance
        query: The vector store query
        clients: V2 client dictionary
        parent: Parent resource path
        **kwargs: Additional arguments

    Returns:
        VectorStoreQueryResult

    """
    vectorsearch = _import_v2_sdk()
    search_client = clients["data_object_search_service_client"]

    # Validate requirements
    if not store.enable_hybrid:
        raise ValueError(
            "HYBRID mode requires enable_hybrid=True in the VertexAIVectorStore "
            "constructor."
        )

    if query.query_embedding is None:
        raise ValueError("HYBRID mode requires query_embedding (dense vector).")

    if query.query_str is None:
        _logger.warning(
            "HYBRID mode without query_str - falling back to vector-only search"
        )
        return _query_v2_default(store, query, clients, parent, **kwargs)

    if store.text_search_fields is None:
        _logger.warning(
            "No text_search_fields configured - falling back to vector-only search"
        )
        return _query_v2_default(store, query, clients, parent, **kwargs)

    # Build filter
    v2_filter = _convert_filters_to_v2(query.filters)

    # Calculate top_k values
    top_k = query.similarity_top_k
    sparse_top_k = query.sparse_top_k or top_k
    hybrid_top_k = query.hybrid_top_k or top_k

    # Build output fields
    output_fields = vectorsearch.OutputFields(
        data_fields=["*"],
        vector_fields=["*"],
        metadata_fields=["*"],
    )

    # Build vector search request
    vector_search_kwargs = {
        "search_field": store.embedding_field,
        "vector": vectorsearch.DenseVector(values=query.query_embedding),
        "top_k": top_k,
        "output_fields": output_fields,
    }
    if v2_filter:
        vector_search_kwargs["filter"] = v2_filter

    # Note: BatchSearchDataObjectsRequest.Search only supports vector_search,
    # not text_search. For true hybrid, we run separate searches and merge.
    try:
        # Run vector search
        vector_request = vectorsearch.SearchDataObjectsRequest(
            parent=parent,
            vector_search=vectorsearch.VectorSearch(**vector_search_kwargs),
        )
        vector_results = list(search_client.search_data_objects(vector_request))

        # Run text search
        text_request = vectorsearch.SearchDataObjectsRequest(
            parent=parent,
            text_search=vectorsearch.TextSearch(
                search_text=query.query_str,
                data_field_names=store.text_search_fields,
                top_k=sparse_top_k,
                output_fields=output_fields,
            ),
        )
        text_results = list(search_client.search_data_objects(text_request))

        # Merge results using RRF
        alpha = query.alpha if query.alpha is not None else store.default_hybrid_alpha
        return _merge_results_rrf(
            store, vector_results, text_results, alpha, hybrid_top_k
        )
    except Exception as e:
        _logger.error(f"Failed to execute HYBRID search: {e}")
        raise


def _query_v2_semantic_hybrid(
    store: Any,
    query: VectorStoreQuery,
    clients: dict,
    parent: str,
    **kwargs: Any,
) -> VectorStoreQueryResult:
    """
    Execute semantic hybrid: VectorSearch + SemanticSearch with ranker.

    Args:
        store: VertexAIVectorStore instance
        query: The vector store query
        clients: V2 client dictionary
        parent: Parent resource path
        **kwargs: Additional arguments

    Returns:
        VectorStoreQueryResult

    """
    vectorsearch = _import_v2_sdk()
    search_client = clients["data_object_search_service_client"]

    if not store.enable_hybrid:
        raise ValueError(
            "SEMANTIC_HYBRID mode requires enable_hybrid=True in the constructor."
        )

    if query.query_str is None:
        raise ValueError("SEMANTIC_HYBRID mode requires query_str.")

    # Calculate top_k values
    top_k = query.similarity_top_k
    hybrid_top_k = query.hybrid_top_k or top_k

    # Build filter
    v2_filter = _convert_filters_to_v2(query.filters)

    # Build output fields
    output_fields = vectorsearch.OutputFields(
        data_fields=["*"],
        vector_fields=["*"],
        metadata_fields=["*"],
    )

    searches = []

    # Add vector search if embedding provided
    if query.query_embedding is not None:
        vector_search_kwargs = {
            "search_field": store.embedding_field,
            "vector": vectorsearch.DenseVector(values=query.query_embedding),
            "top_k": top_k,
            "output_fields": output_fields,
        }
        if v2_filter:
            vector_search_kwargs["filter"] = v2_filter

        searches.append(
            vectorsearch.Search(
                vector_search=vectorsearch.VectorSearch(**vector_search_kwargs)
            )
        )

    # Add semantic search
    searches.append(
        vectorsearch.Search(
            semantic_search=vectorsearch.SemanticSearch(
                search_text=query.query_str,
                search_field=store.embedding_field,
                task_type=store.semantic_task_type,
                top_k=top_k,
                output_fields=output_fields,
            )
        )
    )

    # Build ranker
    ranker = _build_ranker(store, query, num_searches=len(searches))

    # Execute batch search
    batch_request = vectorsearch.BatchSearchDataObjectsRequest(
        parent=parent,
        searches=searches,
        combine=vectorsearch.BatchSearchDataObjectsRequest.CombineResultsOptions(
            ranker=ranker,
            top_k=hybrid_top_k,
            output_fields=output_fields,
        ),
    )

    try:
        results = search_client.batch_search_data_objects(batch_request)
        return _process_batch_search_results(store, results, hybrid_top_k)
    except Exception as e:
        _logger.error(f"Failed to execute SEMANTIC_HYBRID search: {e}")
        raise


# =============================================================================
# Main Query Function with Mode Routing
# =============================================================================


def query_v2(
    store: Any,
    query: VectorStoreQuery,
    **kwargs: Any,
) -> VectorStoreQueryResult:
    """
    Query collection using v2 API with support for multiple query modes.

    Args:
        store: The VertexAIVectorStore instance
        query: The vector store query with mode specification
        **kwargs: Additional keyword arguments

    Returns:
        Vector store query result

    Supported modes:
        - DEFAULT: Dense vector similarity search
        - TEXT_SEARCH: Full-text keyword search
        - HYBRID: Dense vector + text search with RRF/VertexRanker
        - SEMANTIC_HYBRID: Dense vector + semantic search with ranker
        - SPARSE: (Phase 2) Sparse vector search

    """
    _logger.info(
        f"Querying v2 collection: {store.collection_id} with mode: {query.mode}"
    )

    # Get v2 clients
    from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=store.project_id,
        region=store.region,
        credentials_path=store.credentials_path,
    )
    clients = sdk_manager.get_v2_client()
    parent = f"projects/{store.project_id}/locations/{store.region}/collections/{store.collection_id}"

    # Route based on query mode
    if query.mode == VectorStoreQueryMode.DEFAULT:
        return _query_v2_default(store, query, clients, parent, **kwargs)
    elif query.mode == VectorStoreQueryMode.HYBRID:
        return _query_v2_hybrid(store, query, clients, parent, **kwargs)
    elif query.mode == VectorStoreQueryMode.TEXT_SEARCH:
        return _query_v2_text_search(store, query, clients, parent, **kwargs)
    elif query.mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
        return _query_v2_semantic_hybrid(store, query, clients, parent, **kwargs)
    elif query.mode == VectorStoreQueryMode.SPARSE:
        raise NotImplementedError(
            "SPARSE mode is planned for Phase 2 and requires a sparse vector field "
            "configured in the collection schema. Consider using TEXT_SEARCH mode "
            "for keyword search or HYBRID mode for combined vector + keyword search."
        )
    else:
        # Fall back to default for unsupported modes
        _logger.warning(
            f"Query mode {query.mode} not explicitly supported, falling back to DEFAULT"
        )
        return _query_v2_default(store, query, clients, parent, **kwargs)


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
