"""
Vertex AI Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
import os
from typing import Any, List, Optional, Literal, cast

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    MetadataFilters,
)

from llama_index.core.vector_stores.utils import DEFAULT_TEXT_KEY, node_to_metadata_dict
from llama_index.vector_stores.vertexaivectorsearch._sdk_manager import (
    VectorSearchSDKManager,
)
from llama_index.vector_stores.vertexaivectorsearch import utils
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud import storage

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

    text_key: str

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

    # Shared parameters
    batch_size: int = 100
    credentials_path: Optional[str] = None

    _index: MatchingEngineIndex = PrivateAttr()
    _endpoint: MatchingEngineIndexEndpoint = PrivateAttr()
    _index_metadata: dict = PrivateAttr()
    _stream_update: bool = PrivateAttr()
    _staging_bucket: storage.Bucket = PrivateAttr()
    # _document_storage: GCSDocumentStorage = PrivateAttr()

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        gcs_bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        remove_text_from_metadata: bool = True,
        api_version: str = "v1",
        collection_id: Optional[str] = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            project_id=project_id,
            region=region,
            index_id=index_id,
            endpoint_id=endpoint_id,
            gcs_bucket_name=gcs_bucket_name,
            credentials_path=credentials_path,
            text_key=text_key,
            remove_text_from_metadata=remove_text_from_metadata,
            api_version=api_version,
            collection_id=collection_id,
            batch_size=batch_size,
        )

        """Initialize params."""
        # Validate parameters based on API version
        self._validate_parameters()

        # Only initialize v1 resources if using v1 API
        if self.api_version == "v1":
            _sdk_manager = VectorSearchSDKManager(
                project_id=project_id, region=region, credentials_path=credentials_path
            )

            # get index and endpoint resource names including metadata
            self._index = _sdk_manager.get_index(index_id=index_id)
            self._endpoint = _sdk_manager.get_endpoint(endpoint_id=endpoint_id)
            self._index_metadata = self._index.to_dict()

            # get index update method from index metadata
            self._stream_update = False
            if self._index_metadata["indexUpdateMethod"] == "STREAM_UPDATE":
                self._stream_update = True

            # get bucket object when available
            if self.gcs_bucket_name:
                self._staging_bucket = _sdk_manager.get_gcs_bucket(
                    bucket_name=gcs_bucket_name
                )
            else:
                self._staging_bucket = None
        else:
            # v2 initialization will be handled separately
            # Set private attributes to None for now
            self._index = None
            self._endpoint = None
            self._index_metadata = {}
            self._stream_update = False
            self._staging_bucket = None

    def _validate_parameters(self) -> None:
        """Validate parameters based on API version."""
        if self.api_version == "v1":
            # v1 requires index_id and endpoint_id
            if not self.index_id:
                raise ValueError(
                    "index_id is required for v1.0 API. "
                    "Please provide a valid index ID."
                )
            if not self.endpoint_id:
                raise ValueError(
                    "endpoint_id is required for v1.0 API. "
                    "Please provide a valid endpoint ID."
                )
            # v2-exclusive parameters must not be set in v1
            if self.collection_id is not None:
                raise ValueError(
                    "Parameter 'collection_id' is only valid for api_version='v2'. "
                    "For v1, use index_id and endpoint_id instead."
                )
        elif self.api_version == "v2":
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

    @classmethod
    def from_params(
        cls,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        gcs_bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        **kwargs: Any,
    ) -> "VertexAIVectorStore":
        """Create VertexAIVectorStore from config."""
        return cls(
            project_id=project_id,
            region=region,
            index_id=index_id,
            endpoint_id=endpoint_id,
            gcs_bucket_name=gcs_bucket_name,
            credentials_path=credentials_path,
            text_key=text_key,
            api_version="v1",  # Always defaults to v1 for backward compatibility
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "VertexAIVectorStore"

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
        Add nodes to index using v2 API.

        This will be implemented in _v2_operations.py.
        """
        # Import v2 operations module lazily
        from llama_index.vector_stores.vertexaivectorsearch import _v2_operations

        return _v2_operations.add_v2(
            self, nodes, is_complete_overwrite=is_complete_overwrite, **add_kwargs
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            return self._delete_v2(ref_doc_id, **delete_kwargs)
        else:
            return self._delete_v1(ref_doc_id, **delete_kwargs)

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
        Delete nodes using with ref_doc_id (v2 API).

        This will be implemented in _v2_operations.py.
        """
        # Import v2 operations module lazily
        from llama_index.vector_stores.vertexaivectorsearch import _v2_operations

        return _v2_operations.delete_v2(self, ref_doc_id, **delete_kwargs)

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
        """Query index for top k most similar nodes (v2 API)."""
        # Import v2 operations module lazily
        from llama_index.vector_stores.vertexaivectorsearch import _v2_operations

        return _v2_operations.query_v2(self, query, **kwargs)

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
        """Delete nodes by IDs or filters (v2 API)."""
        # Import v2 operations module lazily
        from llama_index.vector_stores.vertexaivectorsearch import _v2_operations

        return _v2_operations.delete_nodes_v2(self, node_ids, filters, **kwargs)

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
        """Clear all nodes from the vector store (v2 API)."""
        # Import v2 operations module lazily
        from llama_index.vector_stores.vertexaivectorsearch import _v2_operations

        return _v2_operations.clear_v2(self)
