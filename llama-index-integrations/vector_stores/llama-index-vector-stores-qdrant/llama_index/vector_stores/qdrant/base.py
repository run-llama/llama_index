"""
Qdrant vector store index.

An index that is built on top of an existing Qdrant collection.

"""

import logging
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import qdrant_client
from qdrant_client import QdrantClient, AsyncQdrantClient
from grpc import RpcError
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.vector_stores.qdrant.utils import (
    HybridFusionCallable,
    SparseEncoderCallable,
    default_sparse_encoder,
    relative_score_fusion,
    fastembed_sparse_encoder,
)
from qdrant_client.conversions.common_types import QuantizationConfig
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.http import models as rest
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchExcept,
    MatchText,
    MatchValue,
    Payload,
    PayloadField,
    Range,
    HasIdCondition,
    IsEmptyCondition,
)
from qdrant_client.qdrant_fastembed import IDF_EMBEDDING_MODELS

logger = logging.getLogger(__name__)
import_err_msg = (
    "`qdrant-client` package not found, please run `pip install qdrant-client`"
)

# Default vector names
DEFAULT_DENSE_VECTOR_NAME = "text-dense"
DEFAULT_SPARSE_VECTOR_NAME_OLD = "text-sparse"
DEFAULT_SPARSE_VECTOR_NAME = "text-sparse-new"
LEGACY_UNNAMED_VECTOR = (
    ""  # The empty string used for unnamed vectors in older collections
)
DOCUMENT_ID_KEY = "doc_id"


class QdrantVectorStore(BasePydanticVectorStore):
    """
    Qdrant Vector Store.

    In this vector store, embeddings and docs are stored within a
    Qdrant collection.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes.

    Args:
        collection_name: (str): name of the Qdrant collection
        client (Optional[QdrantClient]): QdrantClient instance from `qdrant-client` package
        aclient (Optional[AsyncQdrantClient]): AsyncQdrantClient instance from `qdrant-client` package
        url (Optional[str]): url of the Qdrant instance
        api_key (Optional[str]): API key for authenticating with Qdrant
        batch_size (int): number of points to upload in a single request to Qdrant. Defaults to 64
        parallel (int): number of parallel processes to use during upload. Defaults to 1
        max_retries (int): maximum number of retries in case of a failure. Defaults to 3
        client_kwargs (Optional[dict]): additional kwargs for QdrantClient and AsyncQdrantClient
        enable_hybrid (bool): whether to enable hybrid search using dense and sparse vectors
        fastembed_sparse_model (Optional[str]): name of the FastEmbed sparse model to use, if any
        sparse_doc_fn (Optional[SparseEncoderCallable]): function to encode sparse vectors
        sparse_query_fn (Optional[SparseEncoderCallable]): function to encode sparse queries
        hybrid_fusion_fn (Optional[HybridFusionCallable]): function to fuse hybrid search results
        index_doc_id (bool): whether to create a payload index for the document ID. Defaults to True
        text_key (str): Name of the field holding the text information, Defaults to 'text'
        dense_vector_name (Optional[str]): Custom name for the dense vector field. Defaults to 'text-dense'
        sparse_vector_name (Optional[str]): Custom name for the sparse vector field. Defaults to 'text-sparse-new'
        shard_number (Optional[int]): Shard number for sharding the collection
        sharding_method (Optional[rest.ShardingMethod]): Sharding method for the collection
        replication_factor (Optional[int]): Replication factor for the collection
        write_consistency_factor (Optional[int]): Write consistency factor for the collection
        shard_key_selector_fn (Optional[Callable[..., rest.ShardKeySelector]]): Function to select shard keys
        shard_keys (Optional[list[rest.ShardKey]]): List of shard keys
        payload_indexes: Optional[list[dict[str, rest.PayloadSchemaType]]]: List of payload field indexes

    Notes:
        For backward compatibility, the vector store will automatically detect the vector format
        of existing collections and adapt accordingly:
        - For collections created with older versions using unnamed vectors (empty string ""),
          the vector store will use the legacy format for queries.
        - For collections with named vectors, it will use the existing vector names.
        - For new collections, it will use the vector names provided or the defaults.

    Examples:
        `pip install llama-index-vector-stores-qdrant`

        ```python
        import qdrant_client
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        client = qdrant_client.QdrantClient()

        vector_store = QdrantVectorStore(
            collection_name="example_collection", client=client
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = False

    collection_name: str
    url: Optional[str]
    api_key: Optional[str]
    batch_size: int
    parallel: int
    max_retries: int
    client_kwargs: dict = Field(default_factory=dict)
    enable_hybrid: bool
    index_doc_id: bool
    fastembed_sparse_model: Optional[str]
    text_key: Optional[str]
    dense_vector_name: str
    sparse_vector_name: str

    _client: QdrantClient = PrivateAttr()
    _aclient: AsyncQdrantClient = PrivateAttr()
    _collection_initialized: bool = PrivateAttr()
    _sparse_doc_fn: Optional[SparseEncoderCallable] = PrivateAttr()
    _sparse_query_fn: Optional[SparseEncoderCallable] = PrivateAttr()
    _hybrid_fusion_fn: Optional[HybridFusionCallable] = PrivateAttr()
    _dense_config: Optional[rest.VectorParams] = PrivateAttr()
    _sparse_config: Optional[rest.SparseVectorParams] = PrivateAttr()
    _quantization_config: Optional[QuantizationConfig] = PrivateAttr()
    _legacy_vector_format: Optional[bool] = PrivateAttr()
    _shard_key_selector_fn: Optional[Callable[..., rest.ShardKeySelector]] = (
        PrivateAttr()
    )
    _shard_keys: Optional[list[rest.ShardKey]] = PrivateAttr()
    _shard_number: Optional[int] = PrivateAttr()
    _sharding_method: Optional[rest.ShardingMethod] = PrivateAttr()
    _replication_factor: Optional[int] = PrivateAttr()
    _write_consistency_factor: Optional[int] = PrivateAttr()
    _payload_indexes: Optional[list[dict[str, rest.PayloadSchemaType]]] = PrivateAttr()

    def __init__(
        self,
        collection_name: str,
        client: Optional[QdrantClient] = None,
        aclient: Optional[AsyncQdrantClient] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
        client_kwargs: Optional[dict] = None,
        dense_config: Optional[rest.VectorParams] = None,
        sparse_config: Optional[rest.SparseVectorParams] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        enable_hybrid: bool = False,
        fastembed_sparse_model: Optional[str] = None,
        sparse_doc_fn: Optional[SparseEncoderCallable] = None,
        sparse_query_fn: Optional[SparseEncoderCallable] = None,
        hybrid_fusion_fn: Optional[HybridFusionCallable] = None,
        index_doc_id: bool = True,
        text_key: Optional[str] = "text",
        dense_vector_name: Optional[str] = None,
        sparse_vector_name: Optional[str] = None,
        shard_number: Optional[int] = None,
        sharding_method: Optional[rest.ShardingMethod] = None,
        shard_key_selector_fn: Optional[Callable[..., rest.ShardKeySelector]] = None,
        shard_keys: Optional[list[rest.ShardKey]] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        payload_indexes: Optional[list[dict[str, rest.PayloadSchemaType]]] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        # Set default vector names if not provided
        dense_vector_name = dense_vector_name or DEFAULT_DENSE_VECTOR_NAME
        sparse_vector_name = sparse_vector_name or DEFAULT_SPARSE_VECTOR_NAME

        super().__init__(
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            batch_size=batch_size,
            parallel=parallel,
            max_retries=max_retries,
            client_kwargs=client_kwargs or {},
            enable_hybrid=enable_hybrid,
            index_doc_id=index_doc_id,
            fastembed_sparse_model=fastembed_sparse_model,
            text_key=text_key,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
        )

        if (
            client is None
            and aclient is None
            and (url is None or api_key is None or collection_name is None)
        ):
            raise ValueError(
                "Must provide either a QdrantClient instance or a url and api_key."
            )

        if client is None and aclient is None:
            client_kwargs = client_kwargs or {}
            self._client = qdrant_client.QdrantClient(
                url=url, api_key=api_key, **client_kwargs
            )
            self._aclient = qdrant_client.AsyncQdrantClient(
                url=url, api_key=api_key, **client_kwargs
            )
        else:
            if client is not None and aclient is not None:
                possible_local_clients = [
                    getattr(client, "_client", None),
                    getattr(aclient, "_client", None),
                ]
                if any(
                    isinstance(client, QdrantLocal) for client in possible_local_clients
                ):
                    logger.warning(
                        "Both client and aclient are provided. If using `:memory:` "
                        "mode, the data between clients is not synced."
                    )

            self._client = client
            self._aclient = aclient

        self._payload_indexes = payload_indexes

        # Check if collection exists and detect vector format
        self._legacy_vector_format = None
        if self._client is not None:
            self._collection_initialized = self._collection_exists(collection_name)
            if self._collection_initialized:
                self._detect_vector_format(collection_name)
                if self._payload_indexes:
                    self._create_payload_indexes()
        else:
            # Need to do lazy init for async clients
            self._collection_initialized = False

        # Setup hybrid search if enabled
        if enable_hybrid or fastembed_sparse_model is not None:
            enable_hybrid = True
            self._sparse_doc_fn = sparse_doc_fn or self.get_default_sparse_doc_encoder(
                collection_name,
                fastembed_sparse_model=fastembed_sparse_model,
            )
            self._sparse_query_fn = (
                sparse_query_fn
                or self.get_default_sparse_query_encoder(
                    collection_name,
                    fastembed_sparse_model=fastembed_sparse_model,
                )
            )
            self._hybrid_fusion_fn = hybrid_fusion_fn or cast(
                HybridFusionCallable, relative_score_fusion
            )

        self._sparse_config = sparse_config
        self._dense_config = dense_config
        self._quantization_config = quantization_config

        self._shard_number = shard_number
        self._sharding_method = sharding_method
        self._shard_key_selector_fn = shard_key_selector_fn
        self._shard_keys = shard_keys
        self._replication_factor = replication_factor
        self._write_consistency_factor = write_consistency_factor

        if self._sharding_method == rest.ShardingMethod.CUSTOM:
            self._validate_custom_sharding()

    @classmethod
    def class_name(cls) -> str:
        return "QdrantVectorStore"

    def set_query_functions(
        self,
        sparse_doc_fn: Optional[SparseEncoderCallable] = None,
        sparse_query_fn: Optional[SparseEncoderCallable] = None,
        hybrid_fusion_fn: Optional[HybridFusionCallable] = None,
    ):
        self._sparse_doc_fn = sparse_doc_fn
        self._sparse_query_fn = sparse_query_fn
        self._hybrid_fusion_fn = hybrid_fusion_fn

    def _build_points(
        self, nodes: List[BaseNode], sparse_vector_name: str
    ) -> Tuple[List[Any], List[str]]:
        ids = []
        points = []

        for node_batch in iter_batch(nodes, self.batch_size):
            node_ids = []
            vectors: List[Any] = []
            sparse_vectors: List[List[float]] = []
            sparse_indices: List[List[int]] = []
            payloads = []

            if self.enable_hybrid and self._sparse_doc_fn is not None:
                sparse_indices, sparse_vectors = self._sparse_doc_fn(
                    [
                        node.get_content(metadata_mode=MetadataMode.EMBED)
                        for node in node_batch
                    ],
                )

            for i, node in enumerate(node_batch):
                assert isinstance(node, BaseNode)
                node_ids.append(node.node_id)

                if self.enable_hybrid:
                    if (
                        len(sparse_vectors) > 0
                        and len(sparse_indices) > 0
                        and len(sparse_vectors) == len(sparse_indices)
                    ):
                        vectors.append(
                            {
                                # Dynamically switch between the old and new sparse vector name
                                sparse_vector_name: rest.SparseVector(
                                    indices=sparse_indices[i],
                                    values=sparse_vectors[i],
                                ),
                                self.dense_vector_name: node.get_embedding(),
                            }
                        )
                    else:
                        vectors.append(
                            {
                                self.dense_vector_name: node.get_embedding(),
                            }
                        )
                else:
                    vectors.append({self.dense_vector_name: node.get_embedding()})

                metadata = node_to_metadata_dict(
                    node, remove_text=False, flat_metadata=self.flat_metadata
                )

                payloads.append(metadata)

            points.extend(
                [
                    rest.PointStruct(id=node_id, payload=payload, vector=vector)
                    for node_id, payload, vector in zip(node_ids, payloads, vectors)
                ]
            )

            ids.extend(node_ids)

        return points, ids

    def _ensure_async_client(self) -> None:
        if self._aclient is None:
            raise ValueError(
                "Async client is not initialized!\nPlease pass in `aclient` to the constructor: "
                "`QdrantVectorStore(..., aclient=AsyncQdrantClient(...))`"
            )

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        limit: Optional[int] = None,
        shard_identifier: Optional[Any] = None,
    ) -> List[BaseNode]:
        """
        Get nodes from the index.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to retrieve.
            filters (Optional[MetadataFilters]): Metadata filters to apply.
            limit (Optional[int]): Maximum number of nodes to retrieve.
            shard_identifier (Optional[Any]): Shard identifier for the query.

        Returns:
            List[BaseNode]: List of nodes retrieved from the index.

        """
        should = []
        if node_ids is not None:
            should = [
                HasIdCondition(
                    has_id=node_ids,
                )
            ]
            # If we pass a node_ids list,
            # we can limit the search to only those nodes
            # or less if limit is provided
            limit = len(node_ids) if limit is None else min(len(node_ids), limit)

        if filters is not None:
            filter = self._build_subfilter(filters)
            if filter.should is None:
                filter.should = should
            else:
                filter.should.extend(should)
        else:
            filter = Filter(should=should)

        # If we pass an empty list, Qdrant will not return any results
        filter.must = filter.must if filter.must and len(filter.must) > 0 else None
        filter.should = (
            filter.should if filter.should and len(filter.should) > 0 else None
        )
        filter.must_not = (
            filter.must_not if filter.must_not and len(filter.must_not) > 0 else None
        )

        shard_key_selector = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )

        response = self._client.scroll(
            collection_name=self.collection_name,
            limit=limit or 9999,
            scroll_filter=filter,
            with_vectors=True,
            shard_key_selector=shard_key_selector,
        )

        return self.parse_to_query_result(response[0]).nodes

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        limit: Optional[int] = None,
        shard_identifier: Optional[Any] = None,
    ) -> List[BaseNode]:
        """
        Asynchronous method to get nodes from the index.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to retrieve.
            filters (Optional[MetadataFilters]): Metadata filters to apply.
            limit (Optional[int]): Maximum number of nodes to retrieve.
            shard_identifier (Optional[Any]): Shard identifier for the query.

        Returns:
            List[BaseNode]: List of nodes retrieved from the index.

        """
        self._ensure_async_client()

        should = []
        if node_ids is not None:
            should = [
                HasIdCondition(
                    has_id=node_ids,
                )
            ]
            # If we pass a node_ids list,
            # we can limit the search to only those nodes
            # or less if limit is provided
            limit = len(node_ids) if limit is None else min(len(node_ids), limit)

        if filters is not None:
            filter = self._build_subfilter(filters)
            if filter.should is None:
                filter.should = should
            else:
                filter.should.extend(should)
        else:
            filter = Filter(should=should)

        shard_key_selector = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )

        response = await self._aclient.scroll(
            collection_name=self.collection_name,
            limit=limit or 9999,
            scroll_filter=filter,
            with_vectors=True,
            shard_key_selector=shard_key_selector,
        )

        return self.parse_to_query_result(response[0]).nodes

    def add(
        self,
        nodes: List[BaseNode],
        shard_identifier: Optional[Any] = None,
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings
            shard_identifier (Optional[Any]): Shard identifier for the nodes

        """
        if len(nodes) > 0 and not self._collection_initialized:
            self._create_collection(
                collection_name=self.collection_name,
                vector_size=len(nodes[0].get_embedding()),
            )

        if self._collection_initialized and self._legacy_vector_format is None:
            self._detect_vector_format(self.collection_name)

        points, ids = self._build_points(nodes, self.sparse_vector_name)

        shard_key_selector = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )

        self._client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=self.batch_size,
            parallel=self.parallel,
            max_retries=self.max_retries,
            wait=True,
            shard_key_selector=shard_key_selector,
        )

        return ids

    async def async_add(
        self,
        nodes: List[BaseNode],
        shard_identifier: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronous method to add nodes to Qdrant index.

        Args:
            nodes: List[BaseNode]: List of nodes with embeddings.
            shard_identifier: Optional[Any]: Shard identifier for the nodes.

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ValueError: If trying to using async methods without aclient

        """
        from qdrant_client.http.exceptions import UnexpectedResponse

        self._ensure_async_client()

        collection_initialized = await self._acollection_exists(self.collection_name)

        if len(nodes) > 0 and not collection_initialized:
            await self._acreate_collection(
                collection_name=self.collection_name,
                vector_size=len(nodes[0].get_embedding()),
            )
            collection_initialized = True

        if collection_initialized and self._legacy_vector_format is None:
            # If collection exists but we haven't detected the vector format yet
            await self._adetect_vector_format(self.collection_name)

        points, ids = self._build_points(nodes, self.sparse_vector_name)

        shard_key_selector = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )

        for batch in iter_batch(points, self.batch_size):
            retries = 0
            while retries < self.max_retries:
                try:
                    await self._aclient.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        shard_key_selector=shard_key_selector,
                    )
                    break
                except (RpcError, UnexpectedResponse) as exc:
                    retries += 1
                    if retries >= self.max_retries:
                        raise exc  # noqa: TRY201

        return ids

    def delete(
        self,
        ref_doc_id: str,
        shard_identifier: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
            shard_identifier (Optional[Any]): Shard identifier for the nodes.

        """
        shard_key_selector = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key=DOCUMENT_ID_KEY,
                        match=rest.MatchValue(value=ref_doc_id),
                    )
                ]
            ),
            shard_key_selector=shard_key_selector,
        )

    async def adelete(
        self,
        ref_doc_id: str,
        shard_identifier: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Asynchronous method to delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
            shard_identifier (Optional[Any]): Shard identifier for the nodes.

        """
        self._ensure_async_client()

        shard_key_selector = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )

        await self._aclient.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key=DOCUMENT_ID_KEY,
                        match=rest.MatchValue(value=ref_doc_id),
                    )
                ]
            ),
            shard_key_selector=shard_key_selector,
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        shard_identifier: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes using with node_ids.

        Args:
            node_ids (Optional[List[str]): List of node IDs to delete.
            filters (Optional[MetadataFilters]): Metadata filters to apply.
            shard_identifier (Optional[Any]): Shard identifier for the nodes.

        """
        should = []
        if node_ids is not None:
            should = [
                HasIdCondition(
                    has_id=node_ids,
                )
            ]

        if filters is not None:
            filter = self._build_subfilter(filters)
            if filter.should is None:
                filter.should = should
            else:
                filter.should.extend(should)
        else:
            filter = Filter(should=should)

        shard_key_selector = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=filter,
            shard_key_selector=shard_key_selector,
        )

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        shard_identifier: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Asynchronous method to delete nodes using with node_ids.

        Args:
            node_ids (Optional[List[str]): List of node IDs to delete.
            filters (Optional[MetadataFilters]): Metadata filters to apply.
            shard_identifier (Optional[Any]): Shard identifier for the nodes.

        """
        self._ensure_async_client()

        should = []
        if node_ids is not None:
            should = [
                HasIdCondition(
                    has_id=node_ids,
                )
            ]

        if filters is not None:
            filter = self._build_subfilter(filters)
            if filter.should is None:
                filter.should = should
            else:
                filter.should.extend(should)
        else:
            filter = Filter(should=should)

        shard_key_selector = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )

        await self._aclient.delete(
            collection_name=self.collection_name,
            points_selector=filter,
            shard_key_selector=shard_key_selector,
        )

    def clear(self) -> None:
        """
        Clear the index.
        """
        self._client.delete_collection(collection_name=self.collection_name)
        self._collection_initialized = False

    async def aclear(self) -> None:
        """
        Asynchronous method to clear the index.
        """
        self._ensure_async_client()

        await self._aclient.delete_collection(collection_name=self.collection_name)
        self._collection_initialized = False

    @property
    def client(self) -> Any:
        """Return the Qdrant client."""
        return self._client

    def _create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a Qdrant collection."""
        from qdrant_client.http import models as rest
        from qdrant_client.http.exceptions import UnexpectedResponse

        dense_config = self._dense_config or rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        )

        sparse_config = self._sparse_config or rest.SparseVectorParams(
            index=rest.SparseIndexParams(),
            modifier=(
                rest.Modifier.IDF
                if self.fastembed_sparse_model in IDF_EMBEDDING_MODELS
                else None
            ),
        )

        try:
            if self.enable_hybrid:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        self.dense_vector_name: dense_config,
                    },
                    # Newly created collection will have the new sparse vector name
                    sparse_vectors_config={self.sparse_vector_name: sparse_config},
                    quantization_config=self._quantization_config,
                    shard_number=self._shard_number,
                    replication_factor=self._replication_factor,
                    sharding_method=self._sharding_method,
                    write_consistency_factor=self._write_consistency_factor,
                )
            else:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=dense_config,
                    quantization_config=self._quantization_config,
                    shard_number=self._shard_number,
                    replication_factor=self._replication_factor,
                    sharding_method=self._sharding_method,
                    write_consistency_factor=self._write_consistency_factor,
                )

            if self._shard_keys:
                for shard_key in self._shard_keys:
                    self._client.create_shard_key(
                        collection_name=collection_name,
                        shard_key=shard_key,
                    )

            # To improve search performance Qdrant recommends setting up
            # a payload index for fields used in filters.
            # https://qdrant.tech/documentation/concepts/indexing
            if self.index_doc_id:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=DOCUMENT_ID_KEY,
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )

            if self._payload_indexes:
                self._create_payload_indexes()
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            logger.warning(
                "Collection %s already exists, skipping collection creation.",
                collection_name,
            )

            if self._shard_keys:
                for shard_key in self._shard_keys:
                    try:
                        self._client.create_shard_key(
                            collection_name=collection_name,
                            shard_key=shard_key,
                        )
                    except (RpcError, ValueError, UnexpectedResponse) as exc:
                        if "already exists" not in str(exc):
                            raise exc  # noqa: TRY201
                        logger.warning(
                            "Shard key %s already exists, skipping creation.",
                            shard_key,
                        )
                        continue
            if self._payload_indexes:
                self._create_payload_indexes()

        self._collection_initialized = True

    async def _acreate_collection(self, collection_name: str, vector_size: int) -> None:
        """Asynchronous method to create a Qdrant collection."""
        from qdrant_client.http import models as rest
        from qdrant_client.http.exceptions import UnexpectedResponse

        dense_config = self._dense_config or rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        )

        sparse_config = self._sparse_config or rest.SparseVectorParams(
            index=rest.SparseIndexParams(),
            modifier=(
                rest.Modifier.IDF
                if self.fastembed_sparse_model in IDF_EMBEDDING_MODELS
                else None
            ),
        )

        try:
            if self.enable_hybrid:
                await self._aclient.create_collection(
                    collection_name=collection_name,
                    vectors_config={self.dense_vector_name: dense_config},
                    sparse_vectors_config={self.sparse_vector_name: sparse_config},
                    quantization_config=self._quantization_config,
                    shard_number=self._shard_number,
                    replication_factor=self._replication_factor,
                    sharding_method=self._sharding_method,
                    write_consistency_factor=self._write_consistency_factor,
                )
            else:
                await self._aclient.create_collection(
                    collection_name=collection_name,
                    vectors_config=dense_config,
                    quantization_config=self._quantization_config,
                    shard_number=self._shard_number,
                    replication_factor=self._replication_factor,
                    sharding_method=self._sharding_method,
                    write_consistency_factor=self._write_consistency_factor,
                )

            if self._shard_keys:
                for shard_key in self._shard_keys:
                    await self._aclient.create_shard_key(
                        collection_name=collection_name,
                        shard_key=shard_key,
                    )

            # To improve search performance Qdrant recommends setting up
            # a payload index for fields used in filters.
            # https://qdrant.tech/documentation/concepts/indexing
            if self.index_doc_id:
                await self._aclient.create_payload_index(
                    collection_name=collection_name,
                    field_name=DOCUMENT_ID_KEY,
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )

            if self._payload_indexes:
                await self._acreate_payload_indexes()
        except (RpcError, ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            logger.warning(
                "Collection %s already exists, skipping collection creation.",
                collection_name,
            )

            if self._shard_keys:
                for shard_key in self._shard_keys:
                    try:
                        await self._client.create_shard_key(
                            collection_name=collection_name,
                            shard_key=shard_key,
                        )
                    except (RpcError, ValueError, UnexpectedResponse) as exc:
                        if "already exists" not in str(exc):
                            raise exc  # noqa: TRY201
                        logger.warning(
                            "Shard key %s already exists, skipping creation.",
                            shard_key,
                        )
                        continue
            if self._payload_indexes:
                await self._acreate_payload_indexes()

        self._collection_initialized = True

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return self._client.collection_exists(collection_name)

    async def _acollection_exists(self, collection_name: str) -> bool:
        """Asynchronous method to check if a collection exists."""
        return await self._aclient.collection_exists(collection_name)

    def _create_payload_indexes(self) -> None:
        """Create payload indexes in Qdrant collection."""
        if not self._payload_indexes:
            return
        for payload_index in self._payload_indexes:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=payload_index["field_name"],
                field_schema=payload_index["field_schema"],
            )

    async def _acreate_payload_indexes(self) -> None:
        """Create payload indexes in Qdrant collection."""
        if not self._payload_indexes:
            return
        for payload_index in self._payload_indexes:
            await self._aclient.create_payload_index(
                collection_name=self.collection_name,
                field_name=payload_index["field_name"],
                field_schema=payload_index["field_schema"],
            )

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query

        """
        query_embedding = cast(List[float], query.query_embedding)
        #  NOTE: users can pass in qdrant_filters (nested/complicated filters) to override the default MetadataFilters
        qdrant_filters = kwargs.get("qdrant_filters")
        if qdrant_filters is not None:
            query_filter = qdrant_filters
        else:
            query_filter = cast(Filter, self._build_query_filter(query))

        shard_identifier = kwargs.get("shard_identifier")
        shard_key = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None
            else None
        )

        if query.mode == VectorStoreQueryMode.HYBRID and not self.enable_hybrid:
            raise ValueError(
                "Hybrid search is not enabled. Please build the query with "
                "`enable_hybrid=True` in the constructor."
            )
        elif (
            query.mode == VectorStoreQueryMode.HYBRID
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
        ):
            sparse_indices, sparse_embedding = self._sparse_query_fn(
                [query.query_str],
            )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k

            sparse_response = self._client.search_batch(
                collection_name=self.collection_name,
                requests=[
                    rest.SearchRequest(
                        vector=rest.NamedVector(
                            name=self.dense_vector_name,
                            vector=query_embedding,
                        ),
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=True,
                        shard_key=shard_key,
                    ),
                    rest.SearchRequest(
                        vector=rest.NamedSparseVector(
                            name=self.sparse_vector_name,
                            vector=rest.SparseVector(
                                indices=sparse_indices[0],
                                values=sparse_embedding[0],
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
                        shard_key=shard_key,
                    ),
                ],
            )

            # sanity check
            assert len(sparse_response) == 2
            assert self._hybrid_fusion_fn is not None

            # flatten the response
            return self._hybrid_fusion_fn(
                self.parse_to_query_result(sparse_response[0]),
                self.parse_to_query_result(sparse_response[1]),
                # NOTE: only for hybrid search (0 for sparse search, 1 for dense search)
                alpha=query.alpha or 0.5,
                # NOTE: use hybrid_top_k if provided, otherwise use similarity_top_k
                top_k=query.hybrid_top_k or query.similarity_top_k,
            )
        elif (
            query.mode == VectorStoreQueryMode.SPARSE
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
        ):
            sparse_indices, sparse_embedding = self._sparse_query_fn(
                [query.query_str],
            )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k

            sparse_response = self._client.search_batch(
                collection_name=self.collection_name,
                requests=[
                    rest.SearchRequest(
                        vector=rest.NamedSparseVector(
                            name=self.sparse_vector_name,
                            vector=rest.SparseVector(
                                indices=sparse_indices[0],
                                values=sparse_embedding[0],
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
                        shard_key=shard_key,
                    ),
                ],
            )

            return self.parse_to_query_result(sparse_response[0])
        elif self.enable_hybrid:
            # search for dense vectors only
            response = self._client.search_batch(
                collection_name=self.collection_name,
                requests=[
                    rest.SearchRequest(
                        vector=rest.NamedVector(
                            name=self.dense_vector_name,
                            vector=query_embedding,
                        ),
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=True,
                        shard_key=shard_key,
                    ),
                ],
            )

            return self.parse_to_query_result(response[0])
        else:
            # Regular non-hybrid search
            response = self._client.search(
                collection_name=self.collection_name,
                query_vector=(
                    rest.NamedVector(
                        name=self.dense_vector_name,
                        vector=query_embedding,
                    )
                ),
                limit=query.similarity_top_k,
                query_filter=query_filter,
                shard_key_selector=shard_key,
            )
            return self.parse_to_query_result(response)

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Asynchronous method to query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query

        """
        self._ensure_async_client()

        query_embedding = cast(List[float], query.query_embedding)

        #  NOTE: users can pass in qdrant_filters (nested/complicated filters) to override the default MetadataFilters
        qdrant_filters = kwargs.get("qdrant_filters")
        if qdrant_filters is not None:
            query_filter = qdrant_filters
        else:
            # build metadata filters
            query_filter = cast(Filter, self._build_query_filter(query))

        # Check if we need to detect vector format
        if self._legacy_vector_format is None:
            await self._adetect_vector_format(self.collection_name)

        # Get shard_identifier if provided
        shard_identifier = kwargs.get("shard_identifier")
        shard_key = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier
            else None
        )

        if query.mode == VectorStoreQueryMode.HYBRID and not self.enable_hybrid:
            raise ValueError(
                "Hybrid search is not enabled. Please build the query with "
                "`enable_hybrid=True` in the constructor."
            )
        elif (
            query.mode == VectorStoreQueryMode.HYBRID
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
        ):
            sparse_indices, sparse_embedding = self._sparse_query_fn(
                [query.query_str],
            )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k

            sparse_response = await self._aclient.search_batch(
                collection_name=self.collection_name,
                requests=[
                    rest.SearchRequest(
                        vector=rest.NamedVector(
                            name=self.dense_vector_name,
                            vector=query_embedding,
                        ),
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=True,
                        shard_key=shard_key,
                    ),
                    rest.SearchRequest(
                        vector=rest.NamedSparseVector(
                            name=self.sparse_vector_name,
                            vector=rest.SparseVector(
                                indices=sparse_indices[0],
                                values=sparse_embedding[0],
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
                        shard_key=shard_key,
                    ),
                ],
            )

            # sanity check
            assert len(sparse_response) == 2
            assert self._hybrid_fusion_fn is not None

            # flatten the response
            return self._hybrid_fusion_fn(
                self.parse_to_query_result(sparse_response[0]),
                self.parse_to_query_result(sparse_response[1]),
                alpha=query.alpha or 0.5,
                # NOTE: use hybrid_top_k if provided, otherwise use similarity_top_k
                top_k=query.hybrid_top_k or query.similarity_top_k,
            )
        elif (
            query.mode == VectorStoreQueryMode.SPARSE
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
        ):
            sparse_indices, sparse_embedding = self._sparse_query_fn(
                [query.query_str],
            )
            sparse_top_k = query.sparse_top_k or query.similarity_top_k

            sparse_response = await self._aclient.search_batch(
                collection_name=self.collection_name,
                requests=[
                    rest.SearchRequest(
                        vector=rest.NamedSparseVector(
                            name=self.sparse_vector_name,
                            vector=rest.SparseVector(
                                indices=sparse_indices[0],
                                values=sparse_embedding[0],
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
                        shard_key=shard_key,
                    ),
                ],
            )
            return self.parse_to_query_result(sparse_response[0])
        elif self.enable_hybrid:
            # search for dense vectors only
            response = await self._aclient.search_batch(
                collection_name=self.collection_name,
                requests=[
                    rest.SearchRequest(
                        vector=rest.NamedVector(
                            name=self.dense_vector_name,
                            vector=query_embedding,
                        ),
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=True,
                        shard_key=shard_key,
                    ),
                ],
            )

            return self.parse_to_query_result(response[0])
        else:
            response = await self._aclient.search(
                collection_name=self.collection_name,
                query_vector=(
                    rest.NamedVector(
                        name=self.dense_vector_name,
                        vector=query_embedding,
                    )
                ),
                limit=query.similarity_top_k,
                query_filter=query_filter,
                shard_key_selector=shard_key,
            )

            return self.parse_to_query_result(response)

    def parse_to_query_result(self, response: List[Any]) -> VectorStoreQueryResult:
        """
        Convert vector store response to VectorStoreQueryResult.

        Args:
            response: List[Any]: List of results returned from the vector store.

        """
        nodes = []
        similarities = []
        ids = []

        for point in response:
            payload = cast(Payload, point.payload)
            vector = point.vector
            embedding = None

            if isinstance(vector, dict):
                embedding = vector.get(self.dense_vector_name, vector.get("", None))
            elif isinstance(vector, list):
                embedding = vector

            try:
                node = metadata_dict_to_node(payload)

                if embedding and node.embedding is None:
                    node.embedding = embedding
            except Exception:
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    payload
                )

                node = TextNode(
                    id_=str(point.id),
                    text=payload.get(self.text_key),
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                    embedding=embedding,
                )
            nodes.append(node)
            ids.append(str(point.id))
            try:
                similarities.append(point.score)
            except AttributeError:
                # certain requests do not return a score
                similarities.append(1.0)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _build_subfilter(self, filters: MetadataFilters) -> Filter:
        conditions = []
        for subfilter in filters.filters:
            # only for exact match
            if isinstance(subfilter, MetadataFilters) and len(subfilter.filters) > 0:
                conditions.append(self._build_subfilter(subfilter))
            elif not subfilter.operator or subfilter.operator == FilterOperator.EQ:
                if isinstance(subfilter.value, float):
                    conditions.append(
                        FieldCondition(
                            key=subfilter.key,
                            range=Range(
                                gte=subfilter.value,
                                lte=subfilter.value,
                            ),
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=subfilter.key,
                            match=MatchValue(value=subfilter.value),
                        )
                    )
            elif subfilter.operator == FilterOperator.LT:
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(lt=subfilter.value),
                    )
                )
            elif subfilter.operator == FilterOperator.GT:
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(gt=subfilter.value),
                    )
                )
            elif subfilter.operator == FilterOperator.GTE:
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(gte=subfilter.value),
                    )
                )
            elif subfilter.operator == FilterOperator.LTE:
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(lte=subfilter.value),
                    )
                )
            elif (
                subfilter.operator == FilterOperator.TEXT_MATCH
                or subfilter.operator == FilterOperator.TEXT_MATCH_INSENSITIVE
            ):
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchText(text=subfilter.value),
                    )
                )
            elif subfilter.operator == FilterOperator.NE:
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchExcept(**{"except": [subfilter.value]}),
                    )
                )
            elif subfilter.operator == FilterOperator.IN:
                # match any of the values
                # https://qdrant.tech/documentation/concepts/filtering/#match-any
                if isinstance(subfilter.value, List):
                    values = subfilter.value
                else:
                    values = str(subfilter.value).split(",")

                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchAny(any=values),
                    )
                )
            elif subfilter.operator == FilterOperator.NIN:
                # match none of the values
                # https://qdrant.tech/documentation/concepts/filtering/#match-except
                if isinstance(subfilter.value, List):
                    values = subfilter.value
                else:
                    values = str(subfilter.value).split(",")
                conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchExcept(**{"except": values}),
                    )
                )
            elif subfilter.operator == FilterOperator.IS_EMPTY:
                # This condition will match all records where the field reports either does not exist, or has null or [] value.
                # https://qdrant.tech/documentation/concepts/filtering/#is-empty
                conditions.append(
                    IsEmptyCondition(is_empty=PayloadField(key=subfilter.key))
                )

        filter = Filter()
        if filters.condition == FilterCondition.AND:
            filter.must = conditions
        elif filters.condition == FilterCondition.OR:
            filter.should = conditions
        elif filters.condition == FilterCondition.NOT:
            filter.must_not = conditions
        return filter

    def _build_query_filter(self, query: VectorStoreQuery) -> Optional[Any]:
        must_conditions = []

        if query.doc_ids:
            must_conditions.append(
                FieldCondition(
                    key=DOCUMENT_ID_KEY,
                    match=MatchAny(any=query.doc_ids),
                )
            )

        # Point id is a "service" id, it is not stored in payload. There is 'HasId' condition to filter by point id
        # https://qdrant.tech/documentation/concepts/filtering/#has-id
        if query.node_ids:
            must_conditions.append(
                HasIdCondition(has_id=query.node_ids),
            )

        # Qdrant does not use the query.query_str property for the filtering. Full-text
        # filtering cannot handle longer queries and can effectively filter our all the
        # nodes. See: https://github.com/jerryjliu/llama_index/pull/1181

        if query.filters and query.filters.filters:
            must_conditions.append(self._build_subfilter(query.filters))

        if len(must_conditions) == 0:
            return None

        return Filter(must=must_conditions)

    def use_old_sparse_encoder(self, collection_name: str) -> bool:
        """
        Check if the collection uses the old sparse encoder format.
        This is used during initialization to determine which sparse vector name to use.
        """
        collection_exists = self._collection_exists(collection_name)
        if collection_exists:
            cur_collection = self.client.get_collection(collection_name)
            return DEFAULT_SPARSE_VECTOR_NAME_OLD in (
                cur_collection.config.params.sparse_vectors or {}
            )

        return False

    async def ause_old_sparse_encoder(self, collection_name: str) -> bool:
        """
        Asynchronous method to check if the collection uses the old sparse encoder format.
        """
        collection_exists = await self._acollection_exists(collection_name)
        if collection_exists:
            cur_collection = await self._aclient.get_collection(collection_name)
            return DEFAULT_SPARSE_VECTOR_NAME_OLD in (
                cur_collection.config.params.sparse_vectors or {}
            )

        return False

    def get_default_sparse_doc_encoder(
        self,
        collection_name: str,
        fastembed_sparse_model: Optional[str] = None,
    ) -> SparseEncoderCallable:
        """
        Get the default sparse document encoder.
        Use old format for backward compatibility if detected.
        """
        if self.use_old_sparse_encoder(collection_name):
            # Update the sparse vector name to use the old format
            self.sparse_vector_name = DEFAULT_SPARSE_VECTOR_NAME_OLD
            return default_sparse_encoder("naver/efficient-splade-VI-BT-large-doc")

        if fastembed_sparse_model is not None:
            return fastembed_sparse_encoder(model_name=fastembed_sparse_model)

        return fastembed_sparse_encoder()

    def get_default_sparse_query_encoder(
        self,
        collection_name: str,
        fastembed_sparse_model: Optional[str] = None,
    ) -> SparseEncoderCallable:
        """
        Get the default sparse query encoder.
        Use old format for backward compatibility if detected.
        """
        if self.use_old_sparse_encoder(collection_name):
            # Update the sparse vector name to use the old format
            self.sparse_vector_name = DEFAULT_SPARSE_VECTOR_NAME_OLD
            return default_sparse_encoder("naver/efficient-splade-VI-BT-large-query")

        if fastembed_sparse_model is not None:
            return fastembed_sparse_encoder(model_name=fastembed_sparse_model)

        return fastembed_sparse_encoder()

    def _detect_vector_format(self, collection_name: str) -> None:
        """
        Detect the vector format of an existing collection.
        This allows backward compatibility with collections that were created before
        the refactoring to use named vectors consistently.
        """
        try:
            collection_info = self._client.get_collection(collection_name)
            vectors_config = collection_info.config.params.vectors

            # Check if we have an unnamed vector format (where name is empty string)
            if isinstance(vectors_config, dict):
                # Using named vectors format
                if LEGACY_UNNAMED_VECTOR in vectors_config:
                    self._legacy_vector_format = True
                    self.dense_vector_name = LEGACY_UNNAMED_VECTOR
            else:
                # Using unnamed vector format from earlier versions
                self._legacy_vector_format = True
                self.dense_vector_name = LEGACY_UNNAMED_VECTOR

        except Exception as e:
            logger.warning(
                f"Could not detect vector format for collection {collection_name}: {e}"
            )

    async def _adetect_vector_format(self, collection_name: str) -> None:
        """
        Asynchronous method to detect the vector format of an existing collection.
        """
        try:
            collection_info = await self._aclient.get_collection(collection_name)
            vectors_config = collection_info.config.params.vectors

            # Check if we have an unnamed vector format (where name is empty string)
            if isinstance(vectors_config, dict):
                # Using named vectors format
                if LEGACY_UNNAMED_VECTOR in vectors_config:
                    self._legacy_vector_format = True
                    self.dense_vector_name = LEGACY_UNNAMED_VECTOR
            else:
                # Using unnamed vector format from earlier versions
                self._legacy_vector_format = True
                self.dense_vector_name = LEGACY_UNNAMED_VECTOR

        except Exception as e:
            logger.warning(
                f"Could not detect vector format for collection {collection_name}: {e}"
            )

    def _validate_custom_sharding(
        self,
    ):
        """
        Validate custom sharding configuration.
        """
        if not self._shard_key_selector_fn:
            raise ValueError(
                "Must provide a shard_key_selector_fn for custom sharding."
            )
        if not self._shard_keys:
            raise ValueError("Must provide shard_keys for custom sharding.")

    def _generate_shard_key_selector(
        self, shard_identifier: Any
    ) -> Union[rest.ShardKeySelector, None]:
        """
        Generate a shard key selector based on the shard identifier.
        """
        if (
            self._shard_key_selector_fn is not None
            and self._sharding_method == rest.ShardingMethod.CUSTOM
        ):
            return self._shard_key_selector_fn(shard_identifier)

        return None
