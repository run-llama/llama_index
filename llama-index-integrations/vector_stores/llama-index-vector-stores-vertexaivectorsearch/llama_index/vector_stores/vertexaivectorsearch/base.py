"""
Vertex AI Vector store index.

An index that is built on top of an existing vector store.

"""

import asyncio
import logging
import os
import time
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from annotated_types import Ge, Le
from google.api_core.exceptions import NotFound
from google.cloud import storage
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from pydantic import PositiveInt, model_validator
from typing_extensions import Self, override

from llama_index.core.base.embeddings.base_sparse import SparseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    RelatedNodeType,
    TextNode,
)
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
)
from llama_index.vector_stores.vertexaivectorsearch._types import (
    AddBatchResult,
    DeleteBatchResult,
    VertexAIDeleteError,
    VertexAIIndexingError,
    VertexAIInputError,
    VertexAIQueryError,
)
from llama_index.vector_stores.vertexaivectorsearch.utils import (
    convert_filters_to_v2_format,
)

if TYPE_CHECKING:
    from google.cloud.vectorsearch_v1beta import (
        BatchDeleteDataObjectsRequest,
        BatchSearchDataObjectsRequest,
        BatchSearchDataObjectsResponse,
        Collection,
        CreateDataObjectRequest,
        DataObject,
        DataObjectSearchServiceAsyncClient,
        DataObjectSearchServiceClient,
        DataObjectServiceAsyncClient,
        DataObjectServiceClient,
        OutputFields,
        QueryDataObjectsRequest,
        QueryDataObjectsResponse,
        Ranker,
        SearchDataObjectsRequest,
        SearchResult,
        SemanticSearch,
        TextSearch,
        UpdateDataObjectRequest,
        Vector,
        VectorSearch,
        VectorSearchServiceClient,
    )
    from google.cloud.vectorsearch_v1beta.services.data_object_search_service.pagers import (
        QueryDataObjectsAsyncPager,
        QueryDataObjectsPager,
        SearchDataObjectsAsyncPager,
        SearchDataObjectsPager,
    )

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
    # updates not allowed for initialization reasons
    api_version: Literal["v1", "v2"] = Field(default="v1", frozen=True)

    # v1-exclusive parameters
    index_id: str | None = None
    endpoint_id: str | None = None
    gcs_bucket_name: str | None = None

    # v2-exclusive parameters
    collection_id: str | None = None

    # V2 indexing-related fields
    default_add_operation: Literal["create", "update"] = "create"
    max_concurrent_requests: PositiveInt = 5

    # V2 Hybrid Search parameters
    enable_hybrid: bool = False
    text_search_fields: list[str] | None = None

    # V2 fields shared between indexing and query time
    embedding_field: str = "embedding"
    sparse_embedding_field: str = "sparse_embedding"
    # allow users to specify additional fields for V2 indexing (add) operations
    dense_embedding_fields: set[str] = Field(default_factory=set)
    sparse_embedding_fields: set[str] = Field(default_factory=set)

    # optional field names for non-metadata properties in nodes being indexed
    nodeid_field: str | None = None
    node_type_field: str | None = None
    docid_field: str | None = None
    content_field: str | None = None
    content_field_metadata_mode: MetadataMode = MetadataMode.NONE

    # Ranker configuration
    hybrid_ranker: Literal["rrf", "vertex"] = "rrf"
    default_hybrid_alpha: Annotated[float, Ge(0.0), Le(1.0)] = 0.5

    # SemanticSearch configuration
    semantic_task_type: str = "RETRIEVAL_QUERY"
    semantic_search_embedding_field: str | None = None

    # VertexRanker-specific parameters
    vertex_ranker_model: str = Field(default="semantic-ranker-default@latest")
    vertex_ranker_title_field: str | None = None
    vertex_ranker_content_field: str | None = None

    # Shared parameters
    batch_size: PositiveInt = 100
    credentials_path: str | None = None

    # Output field configuration
    # for (a)get_nodes operations
    get_nodes_output_fields: dict[str, list[str]] = Field(
        default={"metadata_fields": ["*"]}
    )
    # for (a)query operations
    query_output_fields: dict[str, list[str]] = Field(
        default={"metadata_fields": ["*"], "data_fields": ["*"]}
    )

    _sdk_manager: VectorSearchSDKManager = PrivateAttr()
    _index: MatchingEngineIndex = PrivateAttr()
    _endpoint: MatchingEngineIndexEndpoint = PrivateAttr()
    _index_metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    _stream_update: bool = PrivateAttr(default=False)
    _staging_bucket: storage.Bucket = PrivateAttr()

    # a semaphore shared across all async_add calls to ensure a global maximum number of
    # simultaneous requests are executed by the same vector store instance
    _async_request_semaphore: asyncio.Semaphore = PrivateAttr()

    @model_validator(mode="after")
    def _validate_embedding_fields(self) -> Self:
        """Validate that embedding fields are not duplicated in metadata."""
        if self.dense_embedding_fields.intersection(self.sparse_embedding_fields):
            raise ValueError(
                f"Field name='{self.embedding_field}' is duplicated in "
                f"'dense_embedding_fields' and 'sparse_embedding_fields'"
            )
        return self

    @override
    def model_post_init(self, context: Any, /) -> None:
        """Validate parameters and initialize any required private attributes."""
        # ensure the SDK manager is created early
        self._sdk_manager = VectorSearchSDKManager(
            project_id=self.project_id,
            region=self.region,
            credentials_path=self.credentials_path,
        )
        # initialize the semaphore with the input value
        self._async_request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

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

            # Hybrid search validation (applies to both v1 and v2, but some features are v2-only)
            if self.enable_hybrid:
                raise ValueError(
                    "enable_hybrid=True is only supported for api_version='v2'. "
                    "V1 hybrid search uses HybridQuery with find_neighbors() directly."
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

    @cached_property
    def v2_vector_search_client(self) -> "VectorSearchServiceClient":
        """Access shared ``VectorSearchServiceClient`` instance."""
        from google.cloud.vectorsearch_v1beta import VectorSearchServiceClient

        credentials = self._sdk_manager.get_v2_credentials()
        return VectorSearchServiceClient(credentials=credentials)

    @cached_property
    def v2_data_object_client(self) -> "DataObjectServiceClient":
        """Access shared ``DataObjectServiceClient`` instance."""
        from google.cloud.vectorsearch_v1beta import DataObjectServiceClient

        credentials = self._sdk_manager.get_v2_credentials()
        return DataObjectServiceClient(credentials=credentials)

    @cached_property
    def v2_data_object_async_client(self) -> "DataObjectServiceAsyncClient":
        """Access shared ``DataObjectServiceAsyncClient`` instance."""
        from google.cloud.vectorsearch_v1beta import DataObjectServiceAsyncClient

        credentials = self._sdk_manager.get_v2_credentials()
        # NOTE: async clients *must* be lazy initialized to ensure the right async loop
        return DataObjectServiceAsyncClient(credentials=credentials)

    @cached_property
    def v2_search_client(self) -> "DataObjectSearchServiceClient":
        """Access shared ``DataObjectSearchServiceClient`` instance."""
        from google.cloud.vectorsearch_v1beta import DataObjectSearchServiceClient

        credentials = self._sdk_manager.get_v2_credentials()
        return DataObjectSearchServiceClient(credentials=credentials)

    @cached_property
    def v2_search_async_client(self) -> "DataObjectSearchServiceAsyncClient":
        """Access shared ``DataObjectSearchServiceAsyncClient`` instance."""
        from google.cloud.vectorsearch_v1beta import DataObjectSearchServiceAsyncClient

        credentials = self._sdk_manager.get_v2_credentials()
        # NOTE: async clients *must* be lazy initialized to ensure the right async loop
        return DataObjectSearchServiceAsyncClient(credentials=credentials)

    @cached_property
    def v2_collection(self) -> "Collection":
        """The V2 collection represented by this vector store instance."""
        self._sdk_manager.ensure_v2_available()
        return self.v2_vector_search_client.get_collection(name=self._collection_parent)

    @cached_property
    def v2_data_fields(self) -> set[str]:
        """The data fields in the V2 collection."""
        # mypy can't infer this type, but we know it's a mapping type
        fields = cast(Mapping, self.v2_collection.data_schema["properties"])
        return set(fields.keys())

    @cached_property
    def v2_vector_fields(self) -> set[str]:
        """The vector fields in the V2 collection."""
        return set(self.v2_collection.vector_schema.keys())

    @cached_property
    def v2_vertex_embedding_fields(self) -> set[str]:
        """
        The set of vector fields in the V2 collection that support Vertex auto-embedding.

        This queries the collection schema once and caches the result. At query-time,
        the semantic search type (i.e., passing a query text for embedding on the Vertex
        server, rather than with a supplied embedding) is only available if the search
        field supports auto-embedding.

        Returns:
            A set of vector fields in the V2 collection that support Vertex auto-embedding.

        """
        return {
            name
            for name, vec_cfg in self.v2_collection.vector_schema.items()
            if (
                vec_cfg.dense_vector
                and vec_cfg.dense_vector.vertex_embedding_config
                and vec_cfg.dense_vector.vertex_embedding_config.model_id
            )
        }

    @override
    def add(
        self,
        nodes: Sequence[BaseNode],
        *,
        is_complete_overwrite: bool = False,
        add_operation: Literal["create", "update"] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes
            is_complete_overwrite: bool:
                (V1 only) whether it is an append or overwrite operation
            add_operation: Literal["create", "update"] | None:
                (V2 only) Specify the operation to be used, overriding ``.add_operation``
            **kwargs: additional keyword arguments to be passed to implementations

        """
        if not nodes:
            _logger.info("Empty node list passed to vector store, not adding")
            return []

        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            if is_complete_overwrite:
                raise VertexAIInputError(
                    "The argument 'is_complete_overwrite' is only valid for api_version='v1'."
                )

            op = add_operation or self.default_add_operation
            _logger.debug(f"Using operation='{op}' for V2 call to 'add'")
            match op:
                case "create":
                    return self._add_v2_create(nodes, **kwargs)
                case "update":
                    return self._add_v2_update(nodes, **kwargs)
                case _:
                    raise VertexAIInputError(f"Unknown add operation: {op}")
        else:
            if add_operation:
                raise VertexAIInputError(
                    "The argument 'add_operation' is only valid for api_version='v2'"
                )
            return self._add_v1(
                nodes, is_complete_overwrite=is_complete_overwrite, **kwargs
            )

    def _add_v1(
        self,
        nodes: Sequence[BaseNode],
        *,
        is_complete_overwrite: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """
        Add nodes to index using v1 API.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings
            is_complete_overwrite: bool: (V1 only) whether it is an append or overwrite operation
            **kwargs: additional keyword arguments (not used)

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
                raise VertexAIInputError(
                    "To update a Vector Search index a staging bucket must be defined."
                )
            utils.batch_update_index(
                index=self._index,
                data_points=data_points,
                staging_bucket=self._staging_bucket,
                is_complete_overwrite=is_complete_overwrite,
            )
        return ids

    def _add_v2_create(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """
        Add nodes to index using the V2 'create' API.

        Args:
            nodes: List[BaseNode]: list of nodes
            **add_kwargs: additional keyword arguments (not used)

        """
        from google.cloud.vectorsearch_v1beta import BatchCreateDataObjectsRequest

        _logger.info(
            f"Adding {len(nodes)} nodes to v2 collection: {self.collection_id}",
        )
        node_ids, batch_requests = self._build_v2_create_requests(nodes)
        result = AddBatchResult()
        time_start = time.perf_counter()
        for i, start in enumerate(range(0, len(batch_requests), self.batch_size)):
            batch = batch_requests[start : start + self.batch_size]
            batch_ids = node_ids[start : start + self.batch_size]
            size = len(batch)
            _logger.info(f"Creating batch {i} ({size} objects)")
            request = BatchCreateDataObjectsRequest(
                parent=self._collection_parent, requests=batch
            )

            try:
                response = self.v2_data_object_client.batch_create_data_objects(request)
                _logger.debug(f"Batch create response: {response}")
                result.added_ids.extend(batch_ids)
                _logger.debug(f"Add request batch {i} complete, indexed {size} nodes")
            except Exception as exc:
                _logger.exception(f"Failed to create batch {i} ({size} objects)")
                result.failed_ids.extend(batch_ids)
                result.exceptions.append(exc)

        time_taken = time.perf_counter() - time_start
        _logger.info(
            f"Completed 'add' operation in {time_taken:.2f}s, {result.summary_line}"
        )
        if not result.succeeded:
            raise VertexAIIndexingError(result)
        return result.added_ids

    def _add_v2_update(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """
        Add nodes to index using the V2 'update' API.

        Args:
            nodes: List[BaseNode]: list of nodes
            **add_kwargs: additional keyword arguments (not used)

        """
        from google.cloud.vectorsearch_v1beta import BatchUpdateDataObjectsRequest

        _logger.info(
            f"Updating {len(nodes)} nodes to v2 collection: {self.collection_id}",
        )
        node_ids, update_requests = self._build_v2_update_requests(nodes)
        result = AddBatchResult()
        time_start = time.perf_counter()
        for i, chunk in enumerate(range(0, len(update_requests), self.batch_size)):
            batch = update_requests[chunk : chunk + self.batch_size]
            batch_ids = node_ids[chunk : chunk + self.batch_size]
            size = len(batch)
            try:
                _logger.info(f"Updating batch {i} ({size} objects)")
                batch_request = BatchUpdateDataObjectsRequest(
                    parent=self._collection_parent, requests=batch
                )
                self.v2_data_object_client.batch_update_data_objects(batch_request)
                result.updated_ids.extend(batch_ids)
                _logger.debug(
                    f"Update request batch {i} complete, indexed {size} nodes"
                )
            except Exception as exc:
                _logger.exception(f"Failed to update batch {i} ({size} objects)")
                result.failed_ids.extend(batch_ids)
                result.exceptions.append(exc)
        time_taken = time.perf_counter() - time_start
        _logger.info(
            f"Completed 'update' operation in {time_taken:.2f}s, {result.summary_line}"
        )
        if not result.succeeded:
            raise VertexAIIndexingError(result)
        return result.updated_ids

    def _build_v2_create_requests(
        self, nodes: Sequence[BaseNode]
    ) -> tuple[list[str], list["CreateDataObjectRequest"]]:
        from google.cloud.vectorsearch_v1beta import CreateDataObjectRequest

        node_ids: list[str] = []
        requests: list[CreateDataObjectRequest] = []
        for node in nodes:
            node_ids.append(node.node_id)
            data_object = self.extract_v2_data_object_from_node(node)
            requests.append(
                CreateDataObjectRequest(
                    parent=self._collection_parent,
                    data_object_id=node.node_id,
                    data_object=data_object,
                )
            )
        return node_ids, requests

    def _build_v2_update_requests(
        self, nodes: Sequence[BaseNode]
    ) -> tuple[list[str], list["UpdateDataObjectRequest"]]:
        from google.cloud.vectorsearch_v1beta import UpdateDataObjectRequest

        node_ids: list[str] = []
        requests: list[UpdateDataObjectRequest] = []
        for node in nodes:
            node_ids.append(node.node_id)
            data_object = self.extract_v2_data_object_from_node(node)
            # updates must include the fully qualified name of the object being updated
            data_object.name = f"{self._collection_parent}/dataObjects/{node.node_id}"
            requests.append(UpdateDataObjectRequest(data_object=data_object))
        return node_ids, requests

    def extract_v2_data_object_from_node(self, node: BaseNode) -> "DataObject":
        """
        Convert a BaseNode to a Vertex ``DataObject`` for adding.

        Handles content, doc ID fields, and dense/sparse embeddings.

        Args:
            node: The llama-index node to convert.

        Returns:
            A ``DataObject`` ready to include in a batch request.

        """
        from google.cloud.vectorsearch_v1beta import (
            DataObject,
            DenseVector,
            SparseVector,
            Vector,
        )

        data: dict[str, Any] = {**node.metadata}
        vectors: dict[str, Vector] = {}

        # node ID field (if not duplicated in metadata)
        if self.nodeid_field:
            data[self.nodeid_field] = node.node_id

        # parent document ID field
        if self.docid_field and node.ref_doc_id:
            data[self.docid_field] = node.ref_doc_id

        # the type of the node
        if self.node_type_field:
            data[self.node_type_field] = node.class_name()

        # node content (e.g., 'node.text' for TextNode)
        if self.content_field:
            data[self.content_field] = node.get_content(
                metadata_mode=self.content_field_metadata_mode
            )

        # dense embeddings
        if node.embedding:
            # special case: embedding stored in node.embedding
            vectors[self.embedding_field] = Vector(
                dense=DenseVector(values=node.get_embedding())
            )
        # all other embeddings are pulled from metadata
        for field_ in self.dense_embedding_fields:
            if field_ in data:
                # remove from the data dict
                vector = data.pop(field_)
                if isinstance(vector, Sequence):
                    vectors[field_] = Vector(dense=DenseVector(values=vector))
                else:
                    _logger.error(
                        f"Invalid dense embedding field '{field_}', type={type(vector)}"
                    )

        # sparse embeddings
        for field_ in self.sparse_embedding_fields.union({self.sparse_embedding_field}):
            if field_ in data:
                # remove from the data dict
                sparse_vector = data.pop(field_)
                if isinstance(sparse_vector, dict):
                    vectors[field_] = Vector(
                        sparse=SparseVector(
                            indices=sparse_vector.get("indices", []),
                            values=sparse_vector.get("values", []),
                        )
                    )
                else:
                    _logger.error(
                        f"Invalid sparse embedding field '{field_}', "
                        f"type={type(sparse_vector)}"
                    )
        return DataObject(data=data, vectors=vectors)

    def extract_node_from_v2_data_object(self, data_obj: "DataObject") -> BaseNode:
        """
        Extract a TextNode and its ID from a Vertex ``DataObject``.

        Args:
            data_obj: A Vertex ``DataObject`` from search results.

        Returns:
            An extracted ``BaseNode``.

            The return type is ``BaseNode`` instead of ``TextNode`` for API compatibility
            reasons, but currently this implementation always returns ``TextNode`` objects.

        """
        # Extract metadata
        metadata: dict[str, Any] = dict(data_obj.data) if data_obj.data else {}

        # Validate node type if required
        if self.node_type_field and self.node_type_field in metadata:
            node_type = metadata.pop(self.node_type_field)
            if node_type != TextNode.class_name():
                raise NotImplementedError(f"Node type '{node_type}' is not supported")

        # Extract node ID from resource, with multiple fallbacks
        if self.nodeid_field and self.nodeid_field in metadata:
            node_id = metadata.pop(self.nodeid_field)
        elif data_obj.data_object_id:
            node_id = data_obj.data_object_id
        elif data_obj.name:
            node_id = Path(data_obj.name).name
        else:  # pragma: no cover
            raise VertexAIInputError(
                f"Input data object has no known ID field: {data_obj}"
            )

        # Extract content if available
        # NOTE: `TextNode` defaults to empty string for this value
        text: str = ""
        if self.content_field and self.content_field in metadata:
            text = metadata.pop(self.content_field)
        elif self.text_key and self.text_key in metadata:
            text = metadata.pop(self.text_key)

        # extract source/parent relationship if configured
        relationships: dict[NodeRelationship, RelatedNodeType] = {}
        if self.docid_field and self.docid_field in metadata:
            parent_id = metadata.pop(self.docid_field)
            relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=parent_id)

        # Extract embeddings
        embedding: list[float] | None = None
        if hasattr(data_obj, "vectors") and data_obj.vectors:
            # special case: the default dense embedding
            if self.embedding_field in data_obj.vectors:
                vector_data = data_obj.vectors[self.embedding_field]
                vector = self._extract_dense_vector(vector_data)
                if vector is not None:
                    embedding = vector
            # extract other known dense vectors
            for dense_field in self.dense_embedding_fields:
                if dense_field in data_obj.vectors:
                    vector_data = data_obj.vectors[dense_field]
                    vector = self._extract_dense_vector(vector_data)
                    if vector is not None:
                        metadata[dense_field] = vector
            # extract any known sparse vectors
            for sparse_field in self.sparse_embedding_fields.union(
                {self.sparse_embedding_field}
            ):
                if sparse_field in data_obj.vectors:
                    sparse_data = data_obj.vectors[sparse_field]
                    sparse_vector = self._extract_sparse_vector(sparse_data)
                    if sparse_vector is not None:
                        metadata[sparse_field] = sparse_vector
        return TextNode(
            id_=node_id,
            text=text,
            metadata=metadata,
            embedding=embedding,
            relationships=relationships,
        )

    @staticmethod
    def _extract_dense_vector(vector: "Vector") -> list[float] | None:
        from google.cloud.vectorsearch_v1beta import DenseVector

        if hasattr(vector, "dense") and isinstance(vector.dense, DenseVector):
            return list(vector.dense.values)
        return None

    @staticmethod
    def _extract_sparse_vector(
        vector: "Vector",
    ) -> dict[str, list[float] | list[int]] | None:
        from google.cloud.vectorsearch_v1beta import SparseVector

        if hasattr(vector, "sparse") and isinstance(vector.sparse, SparseVector):
            return {
                "indices": list(vector.sparse.indices),
                "values": list(vector.sparse.values),
            }
        return None

    @override
    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        *,
        is_complete_overwrite: bool = False,
        add_operation: Literal["create", "update"] | None = None,
        **add_kwargs: Any,
    ) -> list[str]:
        """
        Asynchronously dd nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings
            is_complete_overwrite: bool:
                (V1 only) whether it is an append or overwrite operation
            add_operation: Literal["create", "update"] | None:
                (V2 only) Specify the operation to be used, overriding ``.add_operation``
            **add_kwargs: additional keyword arguments to be passed to implementations

        """
        if not nodes:
            _logger.info("Empty node list passed to vector store, not adding")
            return []

        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            if is_complete_overwrite:
                raise VertexAIInputError(
                    "The argument 'is_complete_overwrite' is only valid for api_version='v1'."
                )

            op = add_operation or self.default_add_operation
            _logger.debug(f"Using operation='{op}' for V2 call to 'async_add'")
            match op:
                case "create":
                    return await self._async_add_v2_create(nodes, **add_kwargs)
                case "update":
                    return await self._async_add_v2_update(nodes, **add_kwargs)
                case _:  # pragma: no cover
                    raise VertexAIInputError(f"Unknown add operation: {op}")
        else:
            # use the synchronous V1 implementation
            if add_operation:
                raise VertexAIInputError(
                    "The argument 'add_operation' is only valid for api_version='v2'"
                )
            return self._add_v1(
                nodes, is_complete_overwrite=is_complete_overwrite, **add_kwargs
            )

    async def _async_add_v2_create(
        self, nodes: Sequence[BaseNode], **add_kwargs: Any
    ) -> list[str]:
        node_ids, requests = self._build_v2_create_requests(nodes)
        tasks = [
            self._async_create_batch(
                batch_idx=i,
                batch_ids=node_ids[start : start + self.batch_size],
                create_requests=requests[start : start + self.batch_size],
            )
            for i, start in enumerate(range(0, len(requests), self.batch_size), start=1)
        ]
        _logger.info(
            f"Async adding {len(nodes)} nodes to v2 collection='{self.collection_id}' in "
            f"{len(tasks)} batches (batch_size={self.batch_size})"
        )

        time_start = time.perf_counter()
        results: list[AddBatchResult] = await asyncio.gather(*tasks)
        time_taken = time.perf_counter() - time_start

        result = sum(results, AddBatchResult())
        _logger.info(
            f"Completed 'async_add' operation in {time_taken:.2f}s, {result.summary_line}"
        )
        if not result.succeeded:
            raise VertexAIIndexingError(result)
        return result.added_ids

    async def _async_create_batch(
        self,
        batch_idx: int,
        batch_ids: list[str],
        create_requests: list["CreateDataObjectRequest"],
    ) -> AddBatchResult:
        """
        Execute an async add for a single batch of data objects.

        Args:
            batch_idx: Index of the batch for logging.
            batch_ids: List of IDs included in the requests.
            create_requests: List of CreateDataObjectRequest protos to create.

        Returns:
            An object containing lists of added and failed IDs.

        """
        from google.cloud.vectorsearch_v1beta import BatchCreateDataObjectsRequest

        # the request will not execute until the semaphore can be acquired
        size = len(create_requests)
        async with self._async_request_semaphore:
            try:
                _logger.debug(f"Adding async batch {batch_idx} ({size} objects)")
                request = BatchCreateDataObjectsRequest(
                    parent=self._collection_parent, requests=create_requests
                )
                response = (
                    await self.v2_data_object_async_client.batch_create_data_objects(
                        request
                    )
                )
                _logger.debug(f"Batch create response: {response}")
                _logger.debug(
                    f"Add request batch {batch_idx} complete, indexed {size} nodes"
                )
                return AddBatchResult(added_ids=batch_ids)
            except Exception as exc:
                _logger.exception(
                    f"Failed to index async batch {batch_idx} ({size} objects)"
                )
                return AddBatchResult(failed_ids=batch_ids, exceptions=[exc])

    async def _async_add_v2_update(
        self, nodes: Sequence[BaseNode], **add_kwargs: Any
    ) -> list[str]:
        """
        Asynchronously update nodes in the collection using v2 API.

        Args:
            nodes: List of nodes
            **add_kwargs: Additional keyword arguments (not used)

        Returns:
            List of node IDs

        """
        node_ids, requests = self._build_v2_update_requests(nodes)
        tasks = [
            self._async_update_batch(
                batch_idx=i,
                batch_ids=node_ids[start : start + self.batch_size],
                update_requests=requests[start : start + self.batch_size],
            )
            for i, start in enumerate(range(0, len(requests), self.batch_size), start=1)
        ]
        _logger.info(
            f"Async updating {len(nodes)} nodes to v2 collection='{self.collection_id}' in "
            f"{len(tasks)} batches (batch_size={self.batch_size})"
        )

        time_start = time.perf_counter()
        results: list[AddBatchResult] = await asyncio.gather(*tasks)
        time_taken = time.perf_counter() - time_start

        result = sum(results, AddBatchResult())
        _logger.info(
            f"Completed 'async_update' operation in {time_taken:.2f}s, {result.summary_line}"
        )
        if not result.succeeded:
            raise VertexAIIndexingError(result)
        return result.updated_ids

    async def _async_update_batch(
        self,
        batch_idx: int,
        batch_ids: list[str],
        update_requests: list["UpdateDataObjectRequest"],
    ) -> AddBatchResult:
        """
        Execute an async update for a single batch of data objects.

        Args:
            batch_idx: Index of the batch for logging.
            batch_ids: List of IDs included in the requests.
            update_requests: List of UpdateDataObjectRequest protos to update.

        Returns:
            An object containing lists of updated and failed IDs.

        """
        from google.cloud.vectorsearch_v1beta import BatchUpdateDataObjectsRequest

        # the request will not execute until the semaphore can be acquired
        size = len(update_requests)
        async with self._async_request_semaphore:
            try:
                _logger.debug(f"Updating async batch {batch_idx} ({size} objects)")
                request = BatchUpdateDataObjectsRequest(
                    parent=self._collection_parent, requests=update_requests
                )
                response = (
                    await self.v2_data_object_async_client.batch_update_data_objects(
                        request
                    )
                )
                _logger.debug(f"Batch update response: {response}")
                _logger.debug(
                    f"Update request batch {batch_idx} complete, updated {size} nodes"
                )
                return AddBatchResult(updated_ids=batch_ids)
            except Exception as exc:
                _logger.exception(
                    f"Failed to update async batch {batch_idx} ({size} objects)"
                )
                return AddBatchResult(failed_ids=batch_ids, exceptions=[exc])

    @override
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
            **delete_kwargs: Additional keyword arguments to be passed to implementations

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            self._delete_v2(ref_doc_id, **delete_kwargs)
        else:
            self._delete_v1(ref_doc_id, **delete_kwargs)

    def _delete_v1(self, ref_doc_id: str, **kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id (v1 API).

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
            **kwargs: Additional keyword arguments (not used)

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
            ref_doc_id: The document ID to delete
            **delete_kwargs: Additional keyword arguments (not used)

        """
        if self.docid_field is None:
            raise VertexAIInputError(
                "The field 'docid_field' must be set to use 'delete'"
            )

        _logger.info(
            f"Deleting nodes with '{self.docid_field}'={ref_doc_id} from v2 collection"
        )
        query_request = self._build_filter_query_request(
            v2_filter={self.docid_field: {"$eq": ref_doc_id}}
        )
        query_results = self.v2_search_client.query_data_objects(query_request)
        time_taken, result = self._sync_delete_query_result(query_results)
        _logger.info(
            f"Delete operation for {self.docid_field}={ref_doc_id} finished in "
            f"{time_taken:.2f}, {result.summary_line}",
        )
        if not result.succeeded:
            raise VertexAIDeleteError(result)

    def _build_filter_query_request(
        self,
        v2_filter: dict[str, Any],
        output_fields: "OutputFields | None" = None,
    ) -> "QueryDataObjectsRequest":
        """Build a query request for a given set of V2 filters."""
        from google.cloud.vectorsearch_v1beta import (
            OutputFields,
            QueryDataObjectsRequest,
        )

        # cannot happen at existing call-sites, but include as a fallback
        if not v2_filter:  # pragma: no cover
            raise VertexAIInputError("Empty filters passed to filter query request")
        if not output_fields:
            output_fields = OutputFields(metadata_fields=["*"])
        return QueryDataObjectsRequest(
            parent=self._collection_parent,
            filter=v2_filter,
            page_size=self.batch_size,
            output_fields=output_fields,
        )

    def _sync_delete_query_result(
        self, paged_resp: "QueryDataObjectsPager"
    ) -> tuple[float, DeleteBatchResult]:
        """Synchronously delete all nodes included in the results of a query."""
        requests = [
            batch_request
            for page in paged_resp.pages
            if (batch_request := self._build_batch_delete_request(page)) is not None
        ]
        _logger.debug(f"Deleting query results in {len(requests)} batches")
        return self._sync_delete_batches(requests)

    def _build_batch_delete_request(
        self, page: "QueryDataObjectsResponse"
    ) -> "BatchDeleteDataObjectsRequest | None":
        from google.cloud.vectorsearch_v1beta import (
            BatchDeleteDataObjectsRequest,
            DeleteDataObjectRequest,
        )

        if delete_requests := [
            DeleteDataObjectRequest(name=obj.name) for obj in page.data_objects
        ]:
            return BatchDeleteDataObjectsRequest(
                parent=self._collection_parent, requests=delete_requests
            )
        return None

    def _sync_delete_batches(
        self, requests: list["BatchDeleteDataObjectsRequest"]
    ) -> tuple[float, DeleteBatchResult]:
        """Execute a set of batch delete requests and output combined results."""
        result = DeleteBatchResult()
        time_start = time.perf_counter()
        for i, batch_request in enumerate(requests):
            size = len(batch_request.requests)
            try:
                _logger.debug(f"Deleting batch {i} ({size} objects)")
                self.v2_data_object_client.batch_delete_data_objects(batch_request)
                result.deleted += size
            except NotFound as exc:
                _logger.warning(
                    f"Delete batch {i} ({size} objects) raised 'NotFound' exception: {exc}"
                )
                result.not_found += size
                result.exceptions.append(exc)
            except Exception as exc:
                _logger.exception(f"Failed to delete batch {i} ({size} objects)")
                result.failed += size
                result.exceptions.append(exc)
        time_taken = time.perf_counter() - time_start
        return time_taken, result

    @override
    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
            **delete_kwargs: Additional keyword arguments

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            await self._adelete_v2(ref_doc_id, **delete_kwargs)
        else:
            # use the sync API for v1
            self._delete_v1(ref_doc_id, **delete_kwargs)

    async def _adelete_v2(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Asynchronously delete nodes using ref_doc_id with v2 API.

        Args:
            ref_doc_id: The document ID to delete
            **delete_kwargs: Additional keyword arguments

        """
        if self.docid_field is None:
            raise VertexAIInputError(
                "The field 'docid_field' must be set to use 'adelete'"
            )

        _logger.info(
            f"Deleting nodes with '{self.docid_field}'={ref_doc_id} from v2 collection"
        )
        query_request = self._build_filter_query_request(
            v2_filter={self.docid_field: {"$eq": ref_doc_id}}
        )
        paged_resp = await self.v2_search_async_client.query_data_objects(query_request)
        time_taken, result = await self._async_delete_query_result(paged_resp)
        _logger.info(
            f"Delete operation for {self.docid_field}={ref_doc_id} finished in "
            f"{time_taken:.2f}, {result.summary_line}",
        )
        if not result.succeeded:
            raise VertexAIDeleteError(result)

    async def _async_delete_query_result(
        self, result_pager: "QueryDataObjectsAsyncPager"
    ) -> tuple[float, DeleteBatchResult]:
        """
        Asynchronously delete all nodes included in the results of a query.

        The paged output from the query is converted into a series of tasks that execute
        asynchronously, respecting the instance-global semaphore controlling
        simultaneous requests to the collection.
        """
        tasks = []
        batch_idx = 1
        async for page in result_pager.pages:
            if batch_request := self._build_batch_delete_request(page):
                tasks.append(self._async_delete_batch(batch_idx, batch_request))
                batch_idx += 1
        time_start = time.perf_counter()
        results: list[DeleteBatchResult] = await asyncio.gather(*tasks)
        time_taken = time.perf_counter() - time_start
        batch_result = sum(results, DeleteBatchResult())
        return time_taken, batch_result

    async def _async_delete_batch(
        self,
        batch_idx: int,
        request: "BatchDeleteDataObjectsRequest",
    ) -> DeleteBatchResult:
        async with self._async_request_semaphore:
            size = len(request.requests)
            try:
                _logger.debug(f"Deleting async batch {batch_idx} ({size} objects)")
                time_start = time.perf_counter()
                await self.v2_data_object_async_client.batch_delete_data_objects(
                    request
                )
                time_taken = time.perf_counter() - time_start
                _logger.debug(
                    f"Delete request batch {batch_idx} complete, deleted {size} nodes "
                    f"in {time_taken:.2f}s",
                )
                return DeleteBatchResult(deleted=size)
            except NotFound as exc:
                _logger.warning(
                    f"Delete batch {batch_idx} ({size} objects) raised 'NotFound' "
                    f"exception: {exc}",
                )
                return DeleteBatchResult(not_found=size, exceptions=[exc])
            except Exception as exc:
                _logger.exception(
                    f"Failed to delete async batch {batch_idx} ({size} objects)"
                )
                return DeleteBatchResult(failed=size, exceptions=[exc])

    @override
    def query(
        self,
        query: VectorStoreQuery,
        *,
        sparse_embedding: SparseEmbedding | None = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Search index for nodes using a specific search mode.

        Supported modes:

        * DEFAULT: Dense vector similarity search via ``query.query_embedding`` or (v2
          only) server-side semantic search. via ``query.query_embedding`` or (V2
          only) server-side semantic search.
        * SPARSE: (V2 only) Sparse vector search
        * TEXT_SEARCH: (V2 only) Full-text keyword search
        * HYBRID: (V2 only) Dense vector + text search with RRF ranker
        * SEMANTIC_HYBRID: (V2 only) Dense vector + semantic search with ranker

        Args:
            query: The vector store query with mode specification
            sparse_embedding:
                (V2 only) A sparse embedding for the query, if query.mode is ``SPARSE``.

                ``VectorStoreQuery`` does not currently have a field for sparse embeddings,
                so it must be supplied as a keyword argument for now.
            **kwargs: Additional keyword arguments

        Returns:
            Vector store query result

        Raises:
            VertexAIQueryError: For invalid ``query`` parameters for the selected mode.

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            return self._query_v2(query, sparse_embedding=sparse_embedding, **kwargs)
        else:
            if sparse_embedding is not None:
                raise VertexAIInputError(
                    "Sparse vector search is not supported in api_version='v1'"
                )
            return self._query_v1(query, **kwargs)

    def _query_v1(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes (v1 API)."""
        query_embedding: list[list[float]] | None = None
        if query.mode == VectorStoreQueryMode.DEFAULT:
            if query.query_embedding:
                query_embedding = [query.query_embedding]
            else:
                raise VertexAIInputError(
                    "query_embedding is required for DEFAULT (vector) search mode. "
                    "Use TEXT_SEARCH mode if you only have a text query."
                )

        if query.filters is not None:
            if "filter" in kwargs and kwargs["filter"] is not None:
                raise VertexAIInputError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for Vertex AI Vector Search specific items that are "
                    "not supported via the generic query interface such as numeric filters."
                )
            filter_, num_filter = utils.to_vectorsearch_filter(query.filters)
        else:
            filter_ = None
            num_filter = None

        matches = utils.find_neighbors(
            index=self._index,
            endpoint=self._endpoint,
            embeddings=cast(list[list[float]], query_embedding),  # validated above
            top_k=query.similarity_top_k,
            filter=filter_,
            numeric_filter=num_filter,
        )

        top_k_nodes: list[BaseNode] = []
        top_k_ids: list[str] = []
        top_k_scores: list[float] = []
        for match in matches:
            node = utils.to_node(match, self.text_key)
            if not match.distance:
                raise ValueError(f"Missing 'distance' field on node={node.node_id}")
            top_k_ids.append(match.id)
            top_k_scores.append(match.distance)
            top_k_nodes.append(node)

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    def _query_v2(
        self,
        query: VectorStoreQuery,
        *,
        sparse_embedding: SparseEmbedding | None = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Search V2 collection for nodes using a specific search mode."""
        from google.cloud.vectorsearch_v1beta import SearchDataObjectsRequest

        _logger.info(
            f"Querying (sync) V2 collection='{self.collection_id}' using mode='{query.mode.value}'"
        )
        # prepare filters and output fields once for all search types
        v2_filter = convert_filters_to_v2_format(query.filters)
        output_fields = self._get_output_fields(query.output_fields)

        # Route based on query mode
        match query.mode:
            case VectorStoreQueryMode.DEFAULT:
                request = self._build_default_request(query, v2_filter, output_fields)
                results = self.v2_search_client.search_data_objects(request)
                return self._process_search_results(results)
            case VectorStoreQueryMode.SPARSE:
                request = SearchDataObjectsRequest(
                    parent=self._collection_parent,
                    vector_search=self._build_sparse_vector_search(
                        query, sparse_embedding, v2_filter, output_fields
                    ),
                )
                results = self.v2_search_client.search_data_objects(request)
                return self._process_search_results(results)
            case VectorStoreQueryMode.TEXT_SEARCH:
                request = SearchDataObjectsRequest(
                    parent=self._collection_parent,
                    text_search=self._build_text_search(
                        query, v2_filter, output_fields
                    ),
                )
                results = self.v2_search_client.search_data_objects(request)
                return self._process_search_results(results)
            case VectorStoreQueryMode.HYBRID:
                batch_request = self._build_hybrid_query_request(
                    query, sparse_embedding, v2_filter, output_fields
                )
                batch_results = self.v2_search_client.batch_search_data_objects(
                    batch_request
                )
                return self._process_batch_search_results(
                    batch_results, batch_request.combine.top_k
                )
            case VectorStoreQueryMode.SEMANTIC_HYBRID:
                batch_request = self._build_semantic_hybrid_query_request(
                    query, sparse_embedding, v2_filter, output_fields
                )
                batch_results = self.v2_search_client.batch_search_data_objects(
                    batch_request
                )
                return self._process_batch_search_results(
                    batch_results, batch_request.combine.top_k
                )
            case _:
                raise VertexAIQueryError(
                    f"Unsupported query mode: {query.mode}, supported modes: "
                    f"DEFAULT, SPARSE, TEXT_SEARCH, HYBRID, SEMANTIC_HYBRID"
                )

    @override
    async def aquery(
        self,
        query: VectorStoreQuery,
        *,
        sparse_embedding: SparseEmbedding | None = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Asynchronously search index for nodes using a specific search mode.

        Supported modes:

        * DEFAULT: Dense vector similarity search via ``query.query_embedding`` or (v2
          only) server-side semantic search.
        * SPARSE: (V2 only) Sparse vector search
        * TEXT_SEARCH: (V2 only) Full-text keyword search
        * HYBRID: (V2 only) Dense vector + text search with RRF ranker
        * SEMANTIC_HYBRID: (V2 only) Dense vector + semantic search with ranker

        Args:
            query: The vector store query with mode specification
            sparse_embedding:
                (V2 only) A sparse embedding for the query, if query.mode is ``SPARSE``.

                ``VectorStoreQuery`` does not currently have a field for sparse embeddings,
                so it must be supplied as a keyword argument for now.
            **kwargs: Additional keyword arguments

        Returns:
            Vector store query result

        Raises:
            VertexAIQueryError: For invalid ``query`` parameters for the selected mode.

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            return await self._aquery_v2(
                query, sparse_embedding=sparse_embedding, **kwargs
            )
        else:
            if sparse_embedding is not None:
                raise VertexAIInputError(
                    "Sparse vector search is not supported in api_version='v1'"
                )
            # use sync version
            return self._query_v1(query, **kwargs)

    async def _aquery_v2(
        self,
        query: VectorStoreQuery,
        *,
        sparse_embedding: SparseEmbedding | None = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Asynchronously search V2 collection for nodes using a specific search mode."""
        from google.cloud.vectorsearch_v1beta import SearchDataObjectsRequest

        _logger.info(
            f"Querying (async) V2 collection='{self.collection_id}' using mode='{query.mode.value}'"
        )
        # prepare filters and output fields once for all search types
        v2_filter = convert_filters_to_v2_format(query.filters)
        output_fields = self._get_output_fields(query.output_fields)

        # Route based on query mode
        match query.mode:
            case VectorStoreQueryMode.DEFAULT:
                request = self._build_default_request(query, v2_filter, output_fields)
                results = await self.v2_search_async_client.search_data_objects(request)
                return await self._process_async_search_results(results)
            case VectorStoreQueryMode.SPARSE:
                request = SearchDataObjectsRequest(
                    parent=self._collection_parent,
                    vector_search=self._build_sparse_vector_search(
                        query, sparse_embedding, v2_filter, output_fields
                    ),
                )
                results = await self.v2_search_async_client.search_data_objects(request)
                return await self._process_async_search_results(results)
            case VectorStoreQueryMode.TEXT_SEARCH:
                request = SearchDataObjectsRequest(
                    parent=self._collection_parent,
                    text_search=self._build_text_search(
                        query, v2_filter, output_fields
                    ),
                )
                results = await self.v2_search_async_client.search_data_objects(request)
                return await self._process_async_search_results(results)
            case VectorStoreQueryMode.HYBRID:
                batch_request = self._build_hybrid_query_request(
                    query, sparse_embedding, v2_filter, output_fields
                )
                batch_results = (
                    await self.v2_search_async_client.batch_search_data_objects(
                        batch_request
                    )
                )
                return self._process_batch_search_results(
                    batch_results, batch_request.combine.top_k
                )
            case VectorStoreQueryMode.SEMANTIC_HYBRID:
                batch_request = self._build_semantic_hybrid_query_request(
                    query, sparse_embedding, v2_filter, output_fields
                )
                batch_results = (
                    await self.v2_search_async_client.batch_search_data_objects(
                        batch_request
                    )
                )
                return self._process_batch_search_results(
                    batch_results, batch_request.combine.top_k
                )
            case _:
                raise VertexAIQueryError(
                    f"Unsupported query mode: {query.mode}, supported modes: "
                    f"DEFAULT, SPARSE, TEXT_SEARCH, HYBRID, SEMANTIC_HYBRID"
                )

    def _get_output_fields(
        self, output_fields: list[str] | None
    ) -> dict[str, list[str]]:
        if not output_fields:
            return self.query_output_fields
        data_fields: list[str] = []
        vector_fields: list[str] = []
        for field_ in set(output_fields):
            if field_ in self.v2_data_fields:
                data_fields.append(field_)
            elif field_ in self.v2_vector_fields:
                vector_fields.append(field_)
            else:
                _logger.warning(
                    f"Unknown field passed in query output fields: {field_}"
                )
        out_fields = {**self.query_output_fields}
        if data_fields:
            out_fields["data_fields"] = sorted(data_fields)
        if vector_fields:
            out_fields["vector_fields"] = sorted(vector_fields)
        return out_fields

    def _build_default_request(
        self,
        query: VectorStoreQuery,
        v2_filter: dict[str, Any] | None,
        output_fields: dict[str, list[str]],
    ) -> "SearchDataObjectsRequest":
        """Build request for either dense (client-side) or semantic (server-side) embedding."""
        from google.cloud.vectorsearch_v1beta import SearchDataObjectsRequest

        if query.query_embedding:
            _logger.debug(f"Using dense search for mode={query.mode.value}")
            request = SearchDataObjectsRequest(
                parent=self._collection_parent,
                vector_search=self._build_dense_vector_search(
                    query, v2_filter, output_fields
                ),
            )
        else:
            _logger.debug(f"Using semantic search for mode={query.mode.value}")
            request = SearchDataObjectsRequest(
                parent=self._collection_parent,
                semantic_search=self._build_semantic_search(
                    query, v2_filter, output_fields
                ),
            )
        return request

    def _build_dense_vector_search(
        self,
        query: VectorStoreQuery,
        v2_filter: dict[str, Any] | None,
        output_fields: dict[str, list[str]],
    ) -> "VectorSearch":
        """
        Build a search object for dense vector similarity search.

        This search type relies on client-side embedding of the query text, and thus
        requires a vector to be passed in the query.

        The field to use can be passed as ``query.embedding_field``; otherwise, the
        default ``.embedding_field`` is used.

        Args:
            query: Query with ``query_embedding`` set.
            v2_filter: V2-compatible filters to be applied to the search, if any.
            output_fields: The fields to return in search results.

        Returns:
            A vector search object to be included in a search request.

        Raises:
            VertexAIQueryError: If ``query_embedding`` is not set.

        """
        from google.cloud.vectorsearch_v1beta import DenseVector, VectorSearch

        if query.query_embedding is None:
            raise VertexAIQueryError(
                "For mode=DEFAULT (dense), 'query_embedding' field must be set"
            )
        search_args: dict[str, Any] = {
            "search_field": query.embedding_field or self.embedding_field,
            "vector": DenseVector(values=query.query_embedding),
            "top_k": query.similarity_top_k,
            "output_fields": output_fields,
        }
        if v2_filter:
            search_args["filter"] = v2_filter
        return VectorSearch(search_args)

    def _build_sparse_vector_search(
        self,
        query: VectorStoreQuery,
        sparse_embedding: SparseEmbedding | None,
        v2_filter: dict[str, Any] | None,
        output_fields: dict[str, list[str]],
    ) -> "VectorSearch":
        """
        Build a search object for sparse vector similarity search.

        This search type relies on client-side embedding of the query text, and thus
        requires a vector to be passed in the query.

        The field to use can be passed as ``query.embedding_field``; otherwise, the
        default ``.sparse_embedding_field`` is used.

        Args:
            query: Query with relevant optional fields set.
            sparse_embedding: Sparse embedding of the query.
            v2_filter: V2-compatible filters to be applied to the search, if any.
            output_fields: The fields to return in search results.

        Returns:
            A vector search object to be included in a search request.

        Raises:
            VertexAIQueryError: If a non-null ``sparse_embedding`` is not passed.

        """
        from google.cloud.vectorsearch_v1beta import SparseVector, VectorSearch

        if not sparse_embedding:
            raise VertexAIQueryError(
                "For mode=SPARSE, a 'sparse_embedding' must be passed"
            )
        search_args: dict[str, Any] = {
            "search_field": query.embedding_field or self.sparse_embedding_field,
            "sparse_vector": SparseVector(
                indices=sparse_embedding.keys(), values=sparse_embedding.values()
            ),
            "top_k": query.sparse_top_k or query.similarity_top_k,
            "output_fields": output_fields,
        }
        if v2_filter:
            search_args["filter"] = v2_filter
        return VectorSearch(search_args)

    def _build_text_search(
        self,
        query: VectorStoreQuery,
        v2_filter: dict[str, Any] | None,
        output_fields: dict[str, list[str]],
    ) -> "TextSearch":
        """
        Build a search object for full-text keyword search.

        Args:
            query: Query with ``query_str`` set.
            v2_filter: V2-compatible filters to be applied to the search, if any.
            output_fields: The fields to return in search results.

        Returns:
            A text search object to be included in a search request.

        Raises:
            VertexAIQueryError:
                If ``query_str`` or ``text_search_fields`` are not configured.

        """
        from google.cloud.vectorsearch_v1beta import TextSearch

        if query.query_str is None:
            raise VertexAIQueryError(
                "For mode=TEXT_SEARCH, 'query_str' field must be set"
            )
        if self.text_search_fields is None:
            raise VertexAIQueryError(
                f"For mode=TEXT_SEARCH, vector store field 'text_search_fields' "
                f"must be set, current={self.text_search_fields}"
            )
        search_args: dict[str, Any] = {
            "search_text": query.query_str,
            "data_field_names": self.text_search_fields,
            "top_k": query.sparse_top_k or query.similarity_top_k,
            "output_fields": output_fields,
        }
        if v2_filter:
            search_args["filter"] = v2_filter
        return TextSearch(search_args)

    def _build_semantic_search(
        self,
        query: VectorStoreQuery,
        v2_filter: dict[str, Any] | None,
        output_fields: dict[str, list[str]],
    ) -> "SemanticSearch":
        """
        Build a search object for semantic similarity search.

        This search type relies on server-side auto-embedding of the query text, so it
        does not use any vectors passed with the input query. The embedding field configured in
        ``.semantic_search_embedding_field`` is used if defined, otherwise the

        To facilitate use with hybrid search (either HYBRID or SEMANTIC_HYBRID), the
        following priority order is used to determine the 'search_field' used:

        1. The store's ``.semantic_search_embedding_field``, if defined
        2. The value of ``query.embedding_field``, if defined and enabled for auto-embedding
        2. The store's of ``.embedding_field``, if defined and enabled for auto-embedding

        Args:
            query: Query with ``query_str`` set.
            v2_filter: V2-compatible filters to be applied to the search, if any.
            output_fields: The fields to return in search results.

        Returns:
            A semantic search object to be included in a search request.

        Raises:
            VertexAIQueryError:
                If ``query.query_str`` is null, or the target embedding field does not
                support auto-embedding.

        """
        from google.cloud.vectorsearch_v1beta import SemanticSearch

        if query.query_str is None:
            raise VertexAIQueryError(
                f"For semantic search with mode={query.mode.value}, the "
                f"'query.query_str' field must be set"
            )

        # determine the appropriate field; complexity is due to requirement support
        # both individual and hybrid semantic searches + limited llama-index API
        search_field: str | None = None
        for field_ in [
            self.semantic_search_embedding_field,
            query.embedding_field,
            self.embedding_field,
        ]:
            if field_ in self.v2_vertex_embedding_fields:
                search_field = field_
                break
        if not search_field:
            raise VertexAIQueryError(
                f"No valid auto-embedding field passed for semantic search, available "
                f"fields: {self.v2_vertex_embedding_fields}"
            )

        search_args: dict[str, Any] = {
            "search_text": query.query_str,
            "search_field": search_field,
            "task_type": self.semantic_task_type,
            "top_k": query.similarity_top_k,
            "output_fields": output_fields,
        }
        if v2_filter:
            search_args["filter"] = v2_filter
        return SemanticSearch(search_args)

    def _build_hybrid_query_request(
        self,
        query: VectorStoreQuery,
        sparse_embedding: SparseEmbedding | None,
        v2_filter: dict[str, Any] | None,
        output_fields: dict[str, list[str]],
    ) -> "BatchSearchDataObjectsRequest":
        """
        Build a query for hybrid search: vector + text search with server-side RRF.

        Text search based on ``query.query_str`` is always used. When
        ``query.query_embedding`` and/or ``sparse_embedding`` are provided, the vector
        search uses all that are provided. Otherwise, this falls back to semantic search
        (server-side auto-embedding) for the vector leg.

        Uses ``BatchSearchDataObjectsRequest`` with ``CombineResultsOptions``
        to perform server-side result fusion.

        Args:
            query:
                Query with ``query_str`` set.

                Optionally include ``query_embedding`` for client-side embedding mode.
            sparse_embedding: Sparse embedding of the query, if needed.
            v2_filter: V2-compatible filters to be applied to the search, if any.
            output_fields: The fields to return in search results.

        Returns:
            A request to execute a hybrid search based on the query.

        Raises:
            VertexAIQueryError:
                If hybrid is not enabled or required fields are missing.

        """
        from google.cloud.vectorsearch_v1beta import (
            BatchSearchDataObjectsRequest,
            Search,
        )

        if not self.enable_hybrid:
            raise VertexAIQueryError(
                f"For mode=HYBRID, vector store field 'enable_hybrid' "
                f"must be True, current={self.enable_hybrid}"
            )
        if query.query_str is None:
            raise VertexAIQueryError("For mode=HYBRID, 'query_str' field must be set")
        if self.text_search_fields is None:
            raise VertexAIQueryError(
                f"For mode=HYBRID, vector store field 'text_search_fields' "
                f"must be set, current={self.text_search_fields}"
            )

        # always use text search
        searches: list[Search] = [
            Search(text_search=self._build_text_search(query, v2_filter, output_fields))
        ]

        # use dense search if an embedding exists
        if query.query_embedding is not None:
            dense_search = self._build_dense_vector_search(
                query, v2_filter, output_fields
            )
            searches.append(Search(vector_search=dense_search))
        # use sparse vector search if the embedding is passed
        if sparse_embedding is not None:
            sparse_search = self._build_sparse_vector_search(
                query, sparse_embedding, v2_filter, output_fields
            )
            searches.append(Search(vector_search=sparse_search))
        # if neither dense nor sparse were passed, fall back to semantic search
        if len(searches) == 1:
            semantic_search = self._build_semantic_search(
                query, v2_filter, output_fields
            )
            searches.append(Search(semantic_search=semantic_search))

        # set up ranker
        top_k = query.hybrid_top_k or query.sparse_top_k or query.similarity_top_k
        ranker = self._build_ranker(query, num_searches=len(searches), top_k=top_k)
        return BatchSearchDataObjectsRequest(
            parent=self._collection_parent,
            searches=searches,
            combine=BatchSearchDataObjectsRequest.CombineResultsOptions(
                ranker=ranker, top_k=top_k, output_fields=output_fields
            ),
        )

    def _build_semantic_hybrid_query_request(
        self,
        query: VectorStoreQuery,
        sparse_embedding: SparseEmbedding | None,
        v2_filter: dict[str, Any] | None,
        output_fields: dict[str, list[str]],
    ) -> "BatchSearchDataObjectsRequest":
        """
        Build a query for semantic hybrid: dense/sparse vector + semantic search with RRF.

        Uses ``BatchSearchDataObjectsRequest`` combining a dense vector search
        with semantic search and server-side RRF ranking.

        Args:
            query:
                Query with ``query_str`` set.

                Optionally include ``query_embedding`` for client-side embedding mode.
            sparse_embedding: Sparse embedding of the query, if needed.
            v2_filter: V2-compatible filters to be applied to the search, if any.
            output_fields: The fields to return in search results.

        Returns:
            A request to execute a semantic hybrid search based on the query.

        Raises:
            VertexAIQueryError:
                If hybrid is not enabled or ``query_str`` is missing.

        """
        from google.cloud.vectorsearch_v1beta import (
            BatchSearchDataObjectsRequest,
            Search,
        )

        if not self.enable_hybrid:
            raise VertexAIQueryError(
                f"For mode=SEMANTIC_HYBRID, vector store field 'enable_hybrid' "
                f"must be True, current={self.enable_hybrid}"
            )
        if query.query_str is None:
            raise VertexAIQueryError(
                "For mode=SEMANTIC_HYBRID, 'query_str' field must be set"
            )

        # always use semantic search
        searches: list[Search] = [
            Search(
                semantic_search=self._build_semantic_search(
                    query, v2_filter, output_fields
                )
            )
        ]
        # use dense search if an embedding exists
        if query.query_embedding is not None:
            searches.append(
                Search(
                    vector_search=self._build_dense_vector_search(
                        query, v2_filter, output_fields
                    )
                )
            )
        # use sparse vector search if the embedding is passed
        if sparse_embedding is not None:
            sparse_search = self._build_sparse_vector_search(
                query, sparse_embedding, v2_filter, output_fields
            )
            searches.append(Search(vector_search=sparse_search))

        # set up ranker
        top_k = query.hybrid_top_k or query.similarity_top_k
        ranker = self._build_ranker(query, num_searches=len(searches), top_k=top_k)
        return BatchSearchDataObjectsRequest(
            parent=self._collection_parent,
            searches=searches,
            combine=BatchSearchDataObjectsRequest.CombineResultsOptions(
                ranker=ranker, top_k=top_k, output_fields=output_fields
            ),
        )

    def _process_search_results(
        self, results: "SearchDataObjectsPager"
    ) -> VectorStoreQueryResult:
        """
        Convert GCP single search results to llama-index VectorStoreQueryResult.

        Args:
            results: Iterable of search results from ``search_data_objects()``.

        Returns:
            A ``VectorStoreQueryResult`` with nodes, similarities, and IDs.

        """
        nodes: list[BaseNode] = []
        ids: list[str] = []
        scores: list[float] = []

        for result in results:
            node = self.extract_node_from_v2_data_object(result.data_object)
            nodes.append(node)
            ids.append(node.id_)
            scores.append(self._extract_score(result))
        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

    async def _process_async_search_results(
        self, results: "SearchDataObjectsAsyncPager"
    ) -> VectorStoreQueryResult:
        """
        Convert GCP single async search results to llama-index VectorStoreQueryResult.

        Args:
            results: Iterable of search results from ``search_data_objects()``.

        Returns:
            A ``VectorStoreQueryResult`` with nodes, similarities, and IDs.

        """
        nodes: list[BaseNode] = []
        ids: list[str] = []
        scores: list[float] = []

        async for result in results:
            node = self.extract_node_from_v2_data_object(result.data_object)
            nodes.append(node)
            ids.append(node.id_)
            scores.append(self._extract_score(result))
        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

    def _process_batch_search_results(
        self, batch_results: "BatchSearchDataObjectsResponse", top_k: int
    ) -> VectorStoreQueryResult:
        """
        Convert GCP batch search (RRF combined) results to llama-index format.

        Handles both ``combined_results`` and ``results`` response structures from
        the ``batch_search_data_objects()`` API.

        Args:
            batch_results: Batch search response from ``batch_search_data_objects()``.
            top_k: Maximum number of results to return.

        Returns:
            A ``VectorStoreQueryResult`` with nodes, similarities, and IDs.

        """
        nodes: list[BaseNode] = []
        ids: list[str] = []
        scores: list[float] = []

        for response in batch_results.results:
            for result in response.results:
                if len(nodes) >= top_k:
                    break
                node = self.extract_node_from_v2_data_object(result.data_object)
                nodes.append(node)
                ids.append(node.id_)
                scores.append(self._extract_score(result))
        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

    @staticmethod
    def _extract_score(result: "SearchResult") -> float:
        """
        Extract the score from a search result object.

        Args:
            result: A single search result from the GCP API.

        Returns:
            The score value (distance, score, rank_score, or 1.0 as fallback).

        """
        if hasattr(result, "distance"):
            return float(result.distance)
        if hasattr(result, "score"):
            return float(result.score)
        if hasattr(result, "rank_score"):
            return float(result.rank_score)
        return 1.0

    def _build_ranker(
        self, query: VectorStoreQuery, num_searches: int, top_k: int
    ) -> "Ranker":
        """
        Build the appropriate ranker based on store configuration.

        Args:
            query: The query being executed
            num_searches: Number of searches being combined
            top_k: Maximum number of results to return.

        Returns:
            Ranker object (RRF only currently)

        """
        from google.cloud.vectorsearch_v1beta import Ranker, ReciprocalRankFusion

        match self.hybrid_ranker:
            case "rrf":
                alpha = (
                    query.alpha
                    if query.alpha is not None
                    else self.default_hybrid_alpha
                )
                weights = self._calculate_rrf_weights(alpha, num_searches)
                return Ranker(rrf=ReciprocalRankFusion(weights=weights))

            case "vertex":
                try:
                    from google.cloud.vectorsearch_v1beta import (  # type: ignore[attr-defined]
                        VertexRanker,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Support for hybrid_ranker='vertex' requires "
                        "'google-cloud-vectorsearch>=0.11.0' and is not available in "
                        "the currently installed version"
                    ) from e

                title_template: str | None = None
                content_template: str | None = None
                if self.vertex_ranker_title_field:
                    title_template = f"{{{{ {self.vertex_ranker_title_field} }}}}"
                if self.vertex_ranker_content_field:
                    content_template = f"{{{{ {self.vertex_ranker_content_field} }}}}"
                return Ranker(
                    vertex=VertexRanker(
                        text_record_spec=VertexRanker.TextRecordSpec(
                            query=query.query_str or "",
                            title_template=title_template,
                            content_template=content_template,
                        ),
                        model=self.vertex_ranker_model,
                        top_n=top_k,
                    )
                )
            case _:  # pragma: no cover
                raise NotImplementedError(
                    f"Unsupported ranker type: {self.hybrid_ranker}"
                )

    @staticmethod
    def _calculate_rrf_weights(alpha: float, num_searches: int = 2) -> list[float]:
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

    @override
    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete nodes by IDs or filters.

        Args:
            node_ids: List of node IDs to delete
            filters: Metadata filters to select nodes for deletion
            **kwargs: Additional keyword arguments

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            self._delete_nodes_v2(node_ids, filters, **kwargs)
        else:
            self._delete_nodes_v1(node_ids, filters, **kwargs)

    def _delete_nodes_v1(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
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
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete nodes by IDs or filters using v2 API.

        Args:
            node_ids: List of node IDs to delete
            filters: Metadata filters to select nodes for deletion
            **kwargs: Additional keyword arguments

        """
        time_taken: float = 0.0
        result = DeleteBatchResult()
        if node_ids:
            batches = self._prepare_delete_by_id_requests(node_ids)
            _logger.info(
                f"Deleting {len(node_ids)} nodes by ID in {len(batches)} batches "
                f"(batch_size={self.batch_size})"
            )
            # execute deletion
            time_taken, result = self._sync_delete_batches(batches)

        elif filters is not None:
            if v2_filter := convert_filters_to_v2_format(filters):
                _logger.info(
                    f"Deleting nodes matching filters={v2_filter} from v2 collection: "
                    f"{self.collection_id}"
                )
                query = self._build_filter_query_request(v2_filter)
                query_results = self.v2_search_client.query_data_objects(query)
                time_taken, result = self._sync_delete_query_result(query_results)
            else:
                _logger.warning(
                    f"Input filter set is empty after conversion, "
                    f"input={filters}, converted={v2_filter}"
                )
        else:
            raise VertexAIInputError("Either node_ids or filters must be provided")

        _logger.info(
            f"Delete for collection finished in {time_taken:.2f}s, {result.summary_line}"
        )
        if not result.succeeded:
            raise VertexAIDeleteError(result)

    def _prepare_delete_by_id_requests(
        self, node_ids: Sequence[str]
    ) -> list["BatchDeleteDataObjectsRequest"]:
        """Batch node IDs into a set of batch requests."""
        from google.cloud.vectorsearch_v1beta import (
            BatchDeleteDataObjectsRequest,
            DeleteDataObjectRequest,
        )

        return [
            BatchDeleteDataObjectsRequest(
                parent=self._collection_parent,
                requests=[
                    DeleteDataObjectRequest(
                        name=f"{self._collection_parent}/dataObjects/{node_id}"
                    )
                    for node_id in node_ids[start : start + self.batch_size]
                ],
            )
            for start in range(0, len(node_ids), self.batch_size)
        ]

    @override
    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete nodes by IDs or filters.

        Args:
            node_ids: List of node IDs to delete
            filters: Metadata filters to select nodes for deletion
            **kwargs: Additional keyword arguments

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            await self._adelete_nodes_v2(node_ids, filters, **kwargs)
        else:
            # use sync for v1
            self._delete_nodes_v1(node_ids, filters, **kwargs)

    async def _adelete_nodes_v2(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Asynchronously delete nodes by IDs or filters using v2 API.

        Args:
            node_ids: List of node IDs to delete
            filters: Metadata filters to select nodes for deletion
            **kwargs: Additional keyword arguments

        """
        # build the appropriate batch request based on input
        time_taken: float = 0.0
        result = DeleteBatchResult()
        if node_ids:
            # batch the calls
            tasks = [
                self._async_delete_batch(i, req)
                for i, req in enumerate(self._prepare_delete_by_id_requests(node_ids))
            ]
            # execute deletion
            _logger.info(
                f"Deleting {len(node_ids)} nodes by ID in {len(tasks)} batches "
                f"(batch_size={self.batch_size})"
            )
            time_start = time.perf_counter()
            results: list[DeleteBatchResult] = await asyncio.gather(*tasks)
            time_taken = time.perf_counter() - time_start
            result = sum(results, DeleteBatchResult())

        elif filters is not None:
            if v2_filter := convert_filters_to_v2_format(filters):
                _logger.info(
                    f"Deleting nodes matching filters (batch_size={self.batch_size}): {v2_filter}"
                )
                query = self._build_filter_query_request(v2_filter)
                query_result = await self.v2_search_async_client.query_data_objects(
                    query
                )
                time_taken, result = await self._async_delete_query_result(query_result)
            else:
                _logger.warning(
                    f"Input filter set is empty after conversion, "
                    f"input={filters}, converted={v2_filter}"
                )
        else:
            raise VertexAIInputError("Either node_ids or filters must be provided")

        _logger.info(
            f"Delete for collection finished in {time_taken:.2f}s, {result.summary_line}"
        )
        if not result.succeeded:
            raise VertexAIDeleteError(result)

    @override
    def clear(self) -> None:
        """Clear all nodes from the vector store."""
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            self._clear_v2()
        else:
            self._clear_v1()

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
        from google.cloud.vectorsearch_v1beta import (
            OutputFields,
            QueryDataObjectsRequest,
        )

        _logger.info(f"Clearing all nodes from v2 collection: {self.collection_id}")
        query_request = QueryDataObjectsRequest(
            parent=self._collection_parent,
            page_size=self.batch_size,
            output_fields=OutputFields(metadata_fields=["*"]),
        )
        paged_resp = self.v2_search_client.query_data_objects(query_request)
        time_taken, result = self._sync_delete_query_result(paged_resp)
        _logger.info(
            f"Clear operation for collection finished in {time_taken:.2f}s, "
            f"{result.summary_line}"
        )
        if not result.succeeded:
            raise VertexAIDeleteError(result)

    @override
    async def aclear(self) -> None:
        """Asynchronously clear all nodes from the vector store."""
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            await self._aclear_v2()
        else:
            # use sync v1 version
            self._clear_v1()

    async def _aclear_v2(self) -> None:
        """
        Asynchronously clear all nodes from the collection using v2 API.
        """
        from google.cloud.vectorsearch_v1beta import (
            OutputFields,
            QueryDataObjectsRequest,
        )

        _logger.info(f"Clearing all nodes from v2 collection: {self.collection_id}")
        query_request = QueryDataObjectsRequest(
            parent=self._collection_parent,
            page_size=self.batch_size,
            output_fields=OutputFields(metadata_fields=["*"]),
        )
        paged_resp = await self.v2_search_async_client.query_data_objects(query_request)
        time_taken, result = await self._async_delete_query_result(paged_resp)
        _logger.info(
            f"Clear operation for collection finished in {time_taken:.2f}s, "
            f"{result.summary_line}"
        )
        if not result.succeeded:
            raise VertexAIDeleteError(result)

    @override
    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        """
        Access nodes from the vector store synchronously by ID or filter results.

        This is not a search operation, nodes are retrieved only by ID or metadata
        filtering. Use :py:meth:`.query` for search operations.

        Args:
            node_ids: The list of node IDs to query.
            filters: A set of filters to apply to the query.

        Returns:
            A list of nodes returned from the store matching in the requested input.

        Raises:
            VertexAIInputError:
                If ``node_ids`` or ``filters`` are either not provided or both provided.

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            return self._get_nodes_v2(node_ids=node_ids, filters=filters)
        else:
            raise NotImplementedError(
                "The 'get_nodes' method is not implemented for v1 stores"
            )

    def _get_nodes_v2(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        """
        Query nodes from the vector store synchronously by ID or filter results.

        This method constructs a ``QueryDataObjectsRequest`` based on the input. For
        ``node_ids``, it uses a filter query for ``object_id`` in the set of node IDs.
        For ``filters``, the filter is converted to the expected format and passed
        directly. If both arguments are passed, and error is raised.

        The output is then converted to :py:class:`~llama_index.core.schema.TextNode`
        object and returned.

        Args:
            node_ids: The list of node IDs to query.
            filters: A set of filters to apply to the query.

        Returns:
            A list of nodes returned from the store matching in the requested input.

        Raises:
            VertexAIInputError:
                If ``node_ids`` or ``filters`` are either not provided or both provided.

        """
        from google.cloud.vectorsearch_v1beta import DataObject, OutputFields

        # build the appropriate filter set based on input
        filter_expr = self._prepare_get_nodes_filters(node_ids, filters)
        if not filter_expr:
            _logger.warning(
                f"Input filter set is empty after conversion, input={filters}"
            )
            return []

        # send the query
        query = self._build_filter_query_request(
            filter_expr, output_fields=OutputFields(**self.get_nodes_output_fields)
        )
        start: float = time.perf_counter()
        query_results = self.v2_search_client.query_data_objects(query)
        time_taken = time.perf_counter() - start

        # load all results and convert to nodes
        data_objects: list[DataObject] = [
            obj for page in query_results.pages for obj in page.data_objects
        ]
        _logger.debug(f"Retrieved {len(data_objects)} data objects")

        nodes = [self.extract_node_from_v2_data_object(obj) for obj in data_objects]
        _logger.info(
            f"Query operation for collection finished in {time_taken:.2f}s, "
            f"output nodes={len(nodes)}"
        )
        return nodes

    @override
    async def aget_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        """
        Access nodes from the vector store asynchronously by ID or filter results.

        This is not a search operation, nodes are retrieved only by ID or metadata
        filtering. Use :py:meth:`.aquery` for search operations.

        Args:
            node_ids: The list of node IDs to query.
            filters: A set of filters to apply to the query.

        Returns:
            A list of nodes returned from the store matching in the requested input.

        Raises:
            VertexAIInputError:
                If ``node_ids`` or ``filters`` are either not provided or both provided.

        """
        if FeatureFlags.should_use_v2(self.api_version):
            # No fallback - v2 requires v2 SDK
            self._sdk_manager.ensure_v2_available()
            return await self._aget_nodes_v2(node_ids=node_ids, filters=filters)
        else:
            raise NotImplementedError(
                "The 'aget_nodes' method is not implemented for v1 stores"
            )

    async def _aget_nodes_v2(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        """
        Query nodes from the vector store asynchronously by ID or filter results.

        This method constructs a ``QueryDataObjectsRequest`` based on the input. For
        ``node_ids``, it uses a filter query for ``object_id`` in the set of node IDs.
        For ``filters``, the filter is converted to the expected format and passed
        directly. If both arguments are passed, and error is raised.

        The output is then converted to :py:class:`~llama_index.core.schema.TextNode`
        object and returned.

        Args:
            node_ids: The list of node IDs to query.
            filters: A set of filters to apply to the query.

        Returns:
            A list of nodes returned from the store matching in the requested input.

        Raises:
            VertexAIInputError:
                If ``node_ids`` or ``filters`` are either not provided or both provided.

        """
        from google.cloud.vectorsearch_v1beta import DataObject, OutputFields

        # build the appropriate filter set based on input
        start: float = time.perf_counter()
        filter_expr = self._prepare_get_nodes_filters(node_ids, filters)
        if not filter_expr:
            _logger.warning(
                f"Input filter set is empty after conversion, input={filters}"
            )
            return []

        # send the query and convert results to nodes
        query = self._build_filter_query_request(
            filter_expr, output_fields=OutputFields(**self.get_nodes_output_fields)
        )
        query_results = await self.v2_search_async_client.query_data_objects(query)
        time_taken = time.perf_counter() - start

        data_objects: list[DataObject] = [
            obj async for page in query_results.pages for obj in page.data_objects
        ]
        _logger.debug(f"Retrieved {len(data_objects)} data objects")

        nodes = [self.extract_node_from_v2_data_object(obj) for obj in data_objects]
        _logger.info(
            f"Query operation for collection finished in {time_taken:.2f}s, "
            f"output nodes={len(nodes)}"
        )
        return nodes

    @staticmethod
    def _prepare_get_nodes_filters(
        node_ids: list[str] | None, filters: MetadataFilters | None
    ) -> dict[str, Any]:
        if node_ids and filters:
            raise VertexAIInputError(
                f"Only one of node_ids or filters may be provided, "
                f"node_ids={node_ids}, filters={filters}"
            )
        if node_ids:
            _logger.info("Querying nodes by ID (%d input IDs)", len(node_ids))
            filter_expr: dict[str, Any] = {"object_id": {"$in": node_ids}}
        elif filters and (v2_filter := convert_filters_to_v2_format(filters)):
            _logger.info("Querying nodes matching filters: %s", v2_filter)
            filter_expr = v2_filter
        elif filters:
            # filter conversion is empty
            _logger.warning(
                "Input filter set is empty after conversion, input=%s", filters
            )
            filter_expr = {}
        else:
            raise VertexAIInputError("Either node_ids or filters must be provided")
        return filter_expr
