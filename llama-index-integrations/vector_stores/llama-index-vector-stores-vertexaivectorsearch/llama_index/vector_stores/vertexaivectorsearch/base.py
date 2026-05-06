"""
Vertex AI Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
import os
from functools import cached_property
from typing import Any, List, Literal, Optional, cast

from google.cloud import storage
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import DEFAULT_TEXT_KEY, node_to_metadata_dict
from llama_index.vector_stores.vertexaivectorsearch import utils
from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
    VectorSearchSDKManager,
    _import_v2_sdk,
)
from llama_index.vector_stores.vertexaivectorsearch.utils import (
    _build_ranker,
    _convert_filters_to_v2,
    _merge_results_rrf,
    _process_batch_search_results,
    _process_search_results,
)
from typing_extensions import override

_logger = logging.getLogger(__name__)


class FeatureFlags:
    """Feature flags for safe v2 rollout."""

    # Environment variable can force v1 even if api_version="v2"
    ENABLE_V2 = os.getenv("VERTEX_AI_ENABLE_V2", "true").lower() == "true"

    @staticmethod
    def should_use_v2(api_version: str) -> bool:
        """Determine if v2 should be used."""
        return api_version == "v2" and FeatureFlags.ENABLE_V2


class VertexAIVectorStore(BasePydanticVectorStore):
    """
    Vertex AI Vector Search vector store.

    In this vector store, embeddings are stored in Vertex AI Vector Store and
    docs are stored within Cloud Storage bucket.

    During query time, the index uses Vertex AI Vector Search to query for the
    top k most similar nodes.

    Args:
        project_id (str) : The Google Cloud Project ID.
        region (str)     : The default location making the API calls.
                           It must be the same location as where Vector Search
                           index created and must be regional.
        index_id (str)   : The fully qualified resource name of the created
                           index in Vertex AI Vector Search.
        endpoint_id (str): The fully qualified resource name of the created
                           index endpoint in Vertex AI Vector Search.
        gcs_bucket_name (Optional[str]):
                           The location where the vectors will be stored for
                           the index to be created in batch mode.
        credentials_path (Optional[str]):
                           The path of the Google credentials on the local file
                           system.

    Examples:
        `pip install llama-index-vector-stores-vertexaivectorsearch`

        ```python
        from
        vector_store = VertexAIVectorStore(
            project_id=PROJECT_ID,
            region=REGION,
            index_id="<index_resource_name>"
            endpoint_id="<index_endpoint_resource_name>"
        )
        ```

    """

    stores_text: bool = True
    remove_text_from_metadata: bool = True
    flat_metadata: bool = False

    text_key: str = DEFAULT_TEXT_KEY

    project_id: str
    region: str

    # API version - defaults to v1 for backward compatibility
    api_version: Literal["v1", "v2"] = Field(default="v1")

    # v1-exclusive parameters
    index_id: Optional[str] = None
    endpoint_id: Optional[str] = None
    gcs_bucket_name: Optional[str] = None

    # v2-exclusive parameters
    collection_id: Optional[str] = None

    # V2 Hybrid Search parameters
    enable_hybrid: bool = Field(default=False)
    text_search_fields: Optional[List[str]] = Field(default=None)
    embedding_field: str = Field(default="embedding")

    # Ranker configuration
    hybrid_ranker: Literal["rrf", "vertex"] = Field(default="rrf")
    default_hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    # SemanticSearch configuration
    semantic_task_type: str = Field(default="RETRIEVAL_QUERY")

    # VertexRanker-specific parameters
    vertex_ranker_model: str = Field(default="semantic-ranker-default@latest")
    vertex_ranker_title_field: Optional[str] = Field(default=None)
    vertex_ranker_content_field: Optional[str] = Field(default=None)

    # Shared parameters
    batch_size: int = 100
    credentials_path: Optional[str] = None

    _sdk_manager: VectorSearchSDKManager = PrivateAttr(default=None)
    _index: MatchingEngineIndex | None = PrivateAttr(default=None)
    _endpoint: MatchingEngineIndexEndpoint | None = PrivateAttr(default=None)
    _index_metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    _stream_update: bool = PrivateAttr(default=False)
    _staging_bucket: storage.Bucket | None = PrivateAttr(default=None)

    # _document_storage: GCSDocumentStorage = PrivateAttr()

    @override
    def model_post_init(self, context: Any, /) -> None:
        """Validate parameters and initialize any required private attributes."""
        self._sdk_manager = VectorSearchSDKManager(
            project_id=self.project_id,
            region=self.region,
            credentials_path=self.credentials_path,
        )

        # V1 validation and initialization
        if self.api_version == "v1":
            # v1 requires index_id and endpoint_id
            if not self.index_id:
                raise ValueError(
                    "index_id is required for v1.0 API. "
                    "Please provide a valid index ID."
                )
            index = self._sdk_manager.get_index(index_id=self.index_id)
            self._index = index
            self._index_metadata = index.to_dict()

            # get index update method from index metadata
            self._stream_update = (
                self._index_metadata["indexUpdateMethod"] == "STREAM_UPDATE"
            )

            if not self.endpoint_id:
                raise ValueError(
                    "endpoint_id is required for v1.0 API. "
                    "Please provide a valid endpoint ID."
                )
            self._endpoint = self._sdk_manager.get_endpoint(
                endpoint_id=self.endpoint_id
            )

            # get bucket object when available
            if self.gcs_bucket_name:
                self._staging_bucket = self._sdk_manager.get_gcs_bucket(
                    bucket_name=self.gcs_bucket_name
                )

            # v2-exclusive parameters must not be set in v1
            if self.collection_id is not None:
                raise ValueError(
                    "Parameter 'collection_id' is only valid for api_version='v2'. "
                    "For v1, use index_id and endpoint_id instead."
                )

        # V2 validation and initialization
        if self.api_version == "v2":
            # v2 requires collection_id
            if not self.collection_id:
                raise ValueError(
                    "collection_id is required for v2.0 API. "
                    "Please provide a valid collection ID."
                )
            # v1-exclusive parameters must not be set in v2
            if self.index_id is not None:
                raise ValueError(
                    "Parameter 'index_id' is only valid for api_version='v1'. "
                    "For v2, use collection_id instead."
                )
            if self.endpoint_id is not None:
                raise ValueError(
                    "Parameter 'endpoint_id' is only valid for api_version='v1'. "
                    "For v2, use collection_id instead."
                )
            if self.gcs_bucket_name is not None:
                raise ValueError(
                    "Parameter 'gcs_bucket_name' is only valid for api_version='v1'. "
                    "v2 does not require a staging bucket."
                )

        # Hybrid search validation (applies to both v1 and v2, but some features are v2-only)
        if self.enable_hybrid and self.api_version != "v2":
            raise ValueError(
                "enable_hybrid=True is only supported for api_version='v2'. "
                "V1 hybrid search uses HybridQuery with find_neighbors() directly."
            )

        if self.hybrid_ranker == "vertex":
            if (
                self.vertex_ranker_title_field is None
                and self.vertex_ranker_content_field is None
            ):
                _logger.warning(
                    "VertexRanker works best with title_field and/or content_field configured. "
                    "Consider setting vertex_ranker_title_field or vertex_ranker_content_field."
                )

    @override
    @classmethod
    def class_name(cls) -> str:
        return "VertexAIVectorStore"

    # V1 properties
    @override
    @property
    def client(self) -> Any:
        """Get client."""
        return self._index

    @property
    def index(self) -> Any:
        """Get client."""
        return self._index

    @property
    def endpoint(self) -> Any:
        """Get client."""
        return self._endpoint

    @property
    def staging_bucket(self) -> Any:
        """Get client."""
        return self._staging_bucket

    # V2 properties
    @cached_property
    def _collection_parent(self) -> str:
        """Full resource path for the collection."""
        return (
            f"projects/{self.project_id}/locations/{self.region}"
            f"/collections/{self.collection_id}"
        )

    @override
    def add(
        self,
        nodes: List[BaseNode],
        is_complete_overwrite: bool = False,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            return self._add_v2(
                nodes, is_complete_overwrite=is_complete_overwrite, **add_kwargs
            )
        else:
            return self._add_v1(
                nodes, is_complete_overwrite=is_complete_overwrite, **add_kwargs
            )

    def _add_v1(
        self,
        nodes: List[BaseNode],
        is_complete_overwrite: bool = False,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index using v1 API.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        ids = []
        embeddings = []
        metadatas = []
        for node in nodes:
            node_id = node.node_id
            metadata = node_to_metadata_dict(
                node, remove_text=False, flat_metadata=False
            )
            embedding = node.get_embedding()

            ids.append(node_id)
            embeddings.append(embedding)
            metadatas.append(metadata)

        data_points = utils.to_data_points(ids, embeddings, metadatas)
        # self._document_storage.add_documents(list(zip(ids, nodes)))

        if self._stream_update:
            utils.stream_update_index(index=self._index, data_points=data_points)
        else:
            if self._staging_bucket is None:
                raise ValueError(
                    "To update a Vector Search index a staging bucket must be defined."
                )
            utils.batch_update_index(
                index=self._index,
                data_points=data_points,
                staging_bucket=self._staging_bucket,
                is_complete_overwrite=is_complete_overwrite,
            )
        return ids

    def _add_v2(
        self,
        nodes: List[BaseNode],
        is_complete_overwrite: bool = False,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to collection using v2 API.

        Args:
            nodes: List of nodes with embeddings
            is_complete_overwrite: Whether to overwrite existing data
            **add_kwargs: Additional keyword arguments

        Returns:
            List of node IDs

        """
        vectorsearch = _import_v2_sdk()

        _logger.info(
            f"Adding {len(nodes)} nodes to v2 collection: {self.collection_id}",
        )

        # Get v2 clients
        clients = self._sdk_manager.get_v2_client()
        data_object_client = clients["data_object_service_client"]

        # Convert nodes to v2 data objects
        batch_requests = []
        ids = []

        for node in nodes:
            node_id = node.node_id
            metadata = node_to_metadata_dict(
                node,
                remove_text=False,
                flat_metadata=False,
            )
            embedding = node.get_embedding()

            # Prepare data and vectors following the notebook pattern
            data_object = {
                "data": metadata,  # Metadata becomes the data field
                "vectors": {
                    # Assuming default embedding field name
                    "embedding": {"dense": {"values": embedding}},
                },
            }

            # Create batch request item
            batch_requests.append(
                {"data_object_id": node_id, "data_object": data_object}
            )
            ids.append(node_id)

        # Batch create data objects
        for i in range(0, len(batch_requests), self.batch_size):
            batch = batch_requests[i : i + self.batch_size]
            _logger.info(
                f"Creating batch {i // self.batch_size + 1} ({len(batch)} objects)"
            )

            request = vectorsearch.BatchCreateDataObjectsRequest(
                parent=self._collection_parent,
                requests=batch,
            )

            try:
                response = data_object_client.batch_create_data_objects(request)
                _logger.debug(f"Batch create response: {response}")
            except Exception as e:
                _logger.error(f"Failed to create data objects batch: {e}")
                raise

        return ids

    @override
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._delete_v2(ref_doc_id, **delete_kwargs)
        else:
            self._delete_v1(ref_doc_id, **delete_kwargs)

    def _delete_v1(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id (v1 API).

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        # get datapoint ids by filter
        filter = {"ref_doc_id": ref_doc_id}
        ids = utils.get_datapoints_by_filter(
            index=self.index, endpoint=self.endpoint, metadata=filter
        )
        # remove datapoints
        self._index.remove_datapoints(datapoint_ids=ids)

    def _delete_v2(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
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
        clients = self._sdk_manager.get_v2_client()
        data_object_client = clients["data_object_service_client"]
        search_client = clients["data_object_search_service_client"]

        # Query for data objects with matching ref_doc_id
        query_request = vectorsearch.QueryDataObjectsRequest(
            parent=self._collection_parent,
            filter={"ref_doc_id": {"$eq": ref_doc_id}},
            output_fields=vectorsearch.OutputFields(
                data_fields=["ref_doc_id"],
                metadata_fields=["*"],
            ),
        )

        try:
            # Execute query
            results = search_client.query_data_objects(query_request)

            # Build batch delete requests
            delete_requests = []
            for data_object in results:
                delete_requests.append(
                    vectorsearch.DeleteDataObjectRequest(name=data_object.name),
                )

            # Batch delete
            if delete_requests:
                batch_delete_request = vectorsearch.BatchDeleteDataObjectsRequest(
                    parent=self._collection_parent,
                    requests=delete_requests,
                )
                response = data_object_client.batch_delete_data_objects(
                    batch_delete_request,
                )
                _logger.info(
                    f"Deleted {len(delete_requests)} data objects with ref_doc_id: {ref_doc_id}",
                )
            else:
                _logger.info(f"No data objects found with ref_doc_id: {ref_doc_id}")
        except Exception as e:
            _logger.error(f"Failed to delete data objects: {e}")
            raise

    @override
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            return self._query_v2(query, **kwargs)
        else:
            return self._query_v1(query, **kwargs)

    def _query_v1(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes (v1 API)."""
        query_embedding = None
        if query.mode == VectorStoreQueryMode.DEFAULT:
            query_embedding = [cast(List[float], query.query_embedding)]

        if query.filters is not None:
            if "filter" in kwargs and kwargs["filter"] is not None:
                raise ValueError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for Vertex AI Vector Search specific items that are "
                    "not supported via the generic query interface such as numeric filters."
                )
            filter, num_filter = utils.to_vectorsearch_filter(query.filters)
        else:
            filter = None
            num_filter = None

        matches = utils.find_neighbors(
            index=self._index,
            endpoint=self._endpoint,
            embeddings=query_embedding,
            top_k=query.similarity_top_k,
            filter=filter,
            numeric_filter=num_filter,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for match in matches:
            node = utils.to_node(match, self.text_key)
            top_k_ids.append(match.id)
            top_k_scores.append(match.distance)
            top_k_nodes.append(node)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    def _query_v2(
        self, query: VectorStoreQuery, **kwargs: Any
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
            f"Querying v2 collection: {self.collection_id} with mode: {query.mode}",
        )

        # Get v2 clients
        clients = self._sdk_manager.get_v2_client()
        # Route based on query mode
        if query.mode == VectorStoreQueryMode.DEFAULT:
            return self._query_v2_default(query, clients, **kwargs)
        elif query.mode == VectorStoreQueryMode.HYBRID:
            return self._query_v2_hybrid(query, clients, **kwargs)
        elif query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            return self._query_v2_text_search(query, clients, **kwargs)
        elif query.mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
            return self._query_v2_semantic_hybrid(query, clients, **kwargs)
        elif query.mode == VectorStoreQueryMode.SPARSE:
            raise NotImplementedError(
                "SPARSE mode is planned for Phase 2 and requires a sparse vector field "
                "configured in the collection schema. Consider using TEXT_SEARCH mode "
                "for keyword search or HYBRID mode for combined vector + keyword search.",
            )
        else:
            # Fall back to default for unsupported modes
            _logger.warning(
                f"Query mode {query.mode} not explicitly supported, falling back to DEFAULT",
            )
            return self._query_v2_default(query, clients, **kwargs)

    def _query_v2_default(
        self,
        query: VectorStoreQuery,
        clients: dict,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Execute default dense vector search.

        Args:
            query: The vector store query
            clients: V2 client dictionary
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
            "parent": self._collection_parent,
            "vector_search": vectorsearch.VectorSearch(
                search_field=self.embedding_field,
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
            return _process_search_results(self, results)
        except Exception as e:
            _logger.error(f"Failed to execute DEFAULT search: {e}")
            raise

    def _query_v2_text_search(
        self,
        query: VectorStoreQuery,
        clients: dict,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Execute full-text keyword search only.

        Args:
            query: The vector store query
            clients: V2 client dictionary
            **kwargs: Additional arguments

        Returns:
            VectorStoreQueryResult

        """
        vectorsearch = _import_v2_sdk()
        search_client = clients["data_object_search_service_client"]

        if query.query_str is None:
            raise ValueError("TEXT_SEARCH mode requires query_str.")

        if self.text_search_fields is None:
            raise ValueError(
                "TEXT_SEARCH mode requires text_search_fields to be configured "
                "in the constructor."
            )

        top_k = query.sparse_top_k or query.similarity_top_k

        # Build search request
        search_request = vectorsearch.SearchDataObjectsRequest(
            parent=self._collection_parent,
            text_search=vectorsearch.TextSearch(
                search_text=query.query_str,
                data_field_names=self.text_search_fields,
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
            return _process_search_results(self, results)
        except Exception as e:
            _logger.error(f"Failed to execute TEXT_SEARCH: {e}")
            raise

    def _query_v2_hybrid(
        self,
        query: VectorStoreQuery,
        clients: dict,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Execute hybrid search: VectorSearch + TextSearch with ranker.

        Args:
            query: The vector store query
            clients: V2 client dictionary
            **kwargs: Additional arguments

        Returns:
            VectorStoreQueryResult

        """
        vectorsearch = _import_v2_sdk()
        search_client = clients["data_object_search_service_client"]

        # Validate requirements
        if not self.enable_hybrid:
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
            return self._query_v2_default(query, clients, **kwargs)

        if self.text_search_fields is None:
            _logger.warning(
                "No text_search_fields configured - falling back to vector-only search"
            )
            return self._query_v2_default(query, clients, **kwargs)

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
            "search_field": self.embedding_field,
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
                parent=self._collection_parent,
                vector_search=vectorsearch.VectorSearch(**vector_search_kwargs),
            )
            vector_results = list(search_client.search_data_objects(vector_request))

            # Run text search
            text_request = vectorsearch.SearchDataObjectsRequest(
                parent=self._collection_parent,
                text_search=vectorsearch.TextSearch(
                    search_text=query.query_str,
                    data_field_names=self.text_search_fields,
                    top_k=sparse_top_k,
                    output_fields=output_fields,
                ),
            )
            text_results = list(search_client.search_data_objects(text_request))

            # Merge results using RRF
            alpha = (
                query.alpha if query.alpha is not None else self.default_hybrid_alpha
            )
            return _merge_results_rrf(
                self, vector_results, text_results, alpha, hybrid_top_k
            )
        except Exception as e:
            _logger.error(f"Failed to execute HYBRID search: {e}")
            raise

    def _query_v2_semantic_hybrid(
        self,
        query: VectorStoreQuery,
        clients: dict,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Execute semantic hybrid: VectorSearch + SemanticSearch with ranker.

        Args:
            query: The vector store query
            clients: V2 client dictionary
            **kwargs: Additional arguments

        Returns:
            VectorStoreQueryResult

        """
        vectorsearch = _import_v2_sdk()
        search_client = clients["data_object_search_service_client"]

        if not self.enable_hybrid:
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
                "search_field": self.embedding_field,
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
                    search_field=self.embedding_field,
                    task_type=self.semantic_task_type,
                    top_k=top_k,
                    output_fields=output_fields,
                )
            )
        )

        # Build ranker
        ranker = _build_ranker(self, query, num_searches=len(searches))

        # Execute batch search
        batch_request = vectorsearch.BatchSearchDataObjectsRequest(
            parent=self._collection_parent,
            searches=searches,
            combine=vectorsearch.BatchSearchDataObjectsRequest.CombineResultsOptions(
                ranker=ranker,
                top_k=hybrid_top_k,
                output_fields=output_fields,
            ),
        )

        try:
            results = search_client.batch_search_data_objects(batch_request)
            return _process_batch_search_results(self, results, hybrid_top_k)
        except Exception as e:
            _logger.error(f"Failed to execute SEMANTIC_HYBRID search: {e}")
            raise

    @override
    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete nodes by IDs or filters.

        Args:
            node_ids: List of node IDs to delete
            filters: Metadata filters to select nodes for deletion

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            return self._delete_nodes_v2(node_ids, filters, **kwargs)
        else:
            return self._delete_nodes_v1(node_ids, filters, **kwargs)

    def _delete_nodes_v1(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> None:
        """Delete nodes by IDs or filters (v1 API)."""
        if node_ids is not None:
            # Delete by node IDs
            self._index.remove_datapoints(datapoint_ids=node_ids)
        else:
            # v1 doesn't have efficient filter-based deletion
            # Would need to query first then delete
            raise NotImplementedError(
                "Filter-based deletion not implemented for v1. "
                "Use delete() with ref_doc_id or provide node_ids."
            )

    def _delete_nodes_v2(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete nodes by IDs or filters using v2 API.

        Args:
            node_ids: List of node IDs to delete
            filters: Metadata filters to select nodes for deletion
            **kwargs: Additional keyword arguments

        """
        vectorsearch = _import_v2_sdk()

        _logger.info(f"Deleting nodes from v2 collection: {self.collection_id}")

        # Get v2 client
        clients = self._sdk_manager.get_v2_client()
        data_object_client = clients["data_object_service_client"]
        search_client = clients["data_object_search_service_client"]

        # Build parent path
        if node_ids is not None:
            # Delete by node IDs
            _logger.info(f"Deleting {len(node_ids)} nodes by ID")

            # Build batch delete requests
            delete_requests = []
            for node_id in node_ids:
                delete_requests.append(
                    vectorsearch.DeleteDataObjectRequest(
                        name=f"{self._collection_parent}/dataObjects/{node_id}",
                    ),
                )

            try:
                if delete_requests:
                    batch_delete_request = vectorsearch.BatchDeleteDataObjectsRequest(
                        parent=self._collection_parent,
                        requests=delete_requests,
                    )
                    response = data_object_client.batch_delete_data_objects(
                        batch_delete_request,
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
                "LlamaIndex MetadataFilters need conversion to v2 filter format.",
            )
            # TODO: Implement filter conversion when we understand the mapping better
            # This would require converting LlamaIndex MetadataFilters to v2's filter format
        else:
            raise ValueError("Either node_ids or filters must be provided")

    @override
    def clear(self) -> None:
        """Clear all nodes from the vector store."""
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            return self._clear_v2()
        else:
            return self._clear_v1()

    def _clear_v1(self) -> None:
        """Clear all nodes from the vector store (v1 API)."""
        raise NotImplementedError(
            "Clear operation not supported in v1. "
            "Please recreate the index or use delete operations."
        )

    def _clear_v2(self) -> None:
        """
        Clear all nodes from the collection using v2 API.
        """
        vectorsearch = _import_v2_sdk()

        _logger.info(f"Clearing all nodes from v2 collection: {self.collection_id}")

        # Get v2 client
        clients = self._sdk_manager.get_v2_client()
        data_object_client = clients["data_object_service_client"]
        search_client = clients["data_object_search_service_client"]

        try:
            # Query all data objects (without filter to get all)
            query_request = vectorsearch.QueryDataObjectsRequest(
                parent=self._collection_parent,
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
                        vectorsearch.DeleteDataObjectRequest(name=data_object.name),
                    )

                # Batch delete this page
                if delete_requests:
                    batch_delete_request = vectorsearch.BatchDeleteDataObjectsRequest(
                        parent=self._collection_parent,
                        requests=delete_requests,
                    )
                    data_object_client.batch_delete_data_objects(batch_delete_request)
                    total_deleted += len(delete_requests)

            _logger.info(f"Cleared {total_deleted} data objects from collection")
        except Exception as e:
            _logger.error(f"Failed to clear collection: {e}")
            raise
