"""
Qdrant vector store index.

An index that is built on top of an existing Qdrant collection.

"""

import logging
from typing import Any, List, Optional, Tuple, cast

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.schema import BaseNode, MetadataMode, TextNode
from llama_index.legacy.utils import iter_batch
from llama_index.legacy.vector_stores.qdrant_utils import (
    HybridFusionCallable,
    SparseEncoderCallable,
    default_sparse_encoder,
    relative_score_fusion,
)
from llama_index.legacy.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.legacy.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)
import_err_msg = (
    "`qdrant-client` package not found, please run `pip install qdrant-client`"
)


class QdrantVectorStore(BasePydanticVectorStore):
    """
    Qdrant Vector Store.

    In this vector store, embeddings and docs are stored within a
    Qdrant collection.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes.

    Args:
        collection_name: (str): name of the Qdrant collection
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
        aclient (Optional[Any]): AsyncQdrantClient instance from `qdrant-client` package
        url (Optional[str]): url of the Qdrant instance
        api_key (Optional[str]): API key for authenticating with Qdrant
        batch_size (int): number of points to upload in a single request to Qdrant. Defaults to 64
        parallel (int): number of parallel processes to use during upload. Defaults to 1
        max_retries (int): maximum number of retries in case of a failure. Defaults to 3
        client_kwargs (Optional[dict]): additional kwargs for QdrantClient and AsyncQdrantClient
        enable_hybrid (bool): whether to enable hybrid search using dense and sparse vectors
        sparse_doc_fn (Optional[SparseEncoderCallable]): function to encode sparse vectors
        sparse_query_fn (Optional[SparseEncoderCallable]): function to encode sparse queries
        hybrid_fusion_fn (Optional[HybridFusionCallable]): function to fuse hybrid search results
    """

    stores_text: bool = True
    flat_metadata: bool = False

    collection_name: str
    path: Optional[str]
    url: Optional[str]
    api_key: Optional[str]
    batch_size: int
    parallel: int
    max_retries: int
    client_kwargs: dict = Field(default_factory=dict)
    enable_hybrid: bool

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()
    _collection_initialized: bool = PrivateAttr()
    _sparse_doc_fn: Optional[SparseEncoderCallable] = PrivateAttr()
    _sparse_query_fn: Optional[SparseEncoderCallable] = PrivateAttr()
    _hybrid_fusion_fn: Optional[HybridFusionCallable] = PrivateAttr()

    def __init__(
        self,
        collection_name: str,
        client: Optional[Any] = None,
        aclient: Optional[Any] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
        client_kwargs: Optional[dict] = None,
        enable_hybrid: bool = False,
        sparse_doc_fn: Optional[SparseEncoderCallable] = None,
        sparse_query_fn: Optional[SparseEncoderCallable] = None,
        hybrid_fusion_fn: Optional[HybridFusionCallable] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            import qdrant_client
        except ImportError:
            raise ImportError(import_err_msg)

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
                logger.warning(
                    "Both client and aclient are provided. If using `:memory:` "
                    "mode, the data between clients is not synced."
                )

            self._client = client
            self._aclient = aclient

        if self._client is not None:
            self._collection_initialized = self._collection_exists(collection_name)
        else:
            #  need to do lazy init for async clients
            self._collection_initialized = False

        # setup hybrid search if enabled
        if enable_hybrid:
            self._sparse_doc_fn = sparse_doc_fn or default_sparse_encoder(
                "naver/efficient-splade-VI-BT-large-doc"
            )
            self._sparse_query_fn = sparse_query_fn or default_sparse_encoder(
                "naver/efficient-splade-VI-BT-large-query"
            )
            self._hybrid_fusion_fn = hybrid_fusion_fn or cast(
                HybridFusionCallable, relative_score_fusion
            )

        super().__init__(
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            batch_size=batch_size,
            parallel=parallel,
            max_retries=max_retries,
            client_kwargs=client_kwargs or {},
            enable_hybrid=enable_hybrid,
        )

    @classmethod
    def class_name(cls) -> str:
        return "QdrantVectorStore"

    def _build_points(self, nodes: List[BaseNode]) -> Tuple[List[Any], List[str]]:
        from qdrant_client.http import models as rest

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
                                "text-sparse": rest.SparseVector(
                                    indices=sparse_indices[i],
                                    values=sparse_vectors[i],
                                ),
                                "text-dense": node.get_embedding(),
                            }
                        )
                    else:
                        vectors.append(
                            {
                                "text-dense": node.get_embedding(),
                            }
                        )
                else:
                    vectors.append(node.get_embedding())

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

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if len(nodes) > 0 and not self._collection_initialized:
            self._create_collection(
                collection_name=self.collection_name,
                vector_size=len(nodes[0].get_embedding()),
            )

        points, ids = self._build_points(nodes)

        self._client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=self.batch_size,
            parallel=self.parallel,
            max_retries=self.max_retries,
            wait=True,
        )

        return ids

    async def async_add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """
        Asynchronous method to add nodes to Qdrant index.

        Args:
            nodes: List[BaseNode]: List of nodes with embeddings.

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ValueError: If trying to using async methods without aclient
        """
        collection_initialized = await self._acollection_exists(self.collection_name)

        if len(nodes) > 0 and not collection_initialized:
            await self._acreate_collection(
                collection_name=self.collection_name,
                vector_size=len(nodes[0].get_embedding()),
            )

        points, ids = self._build_points(nodes)

        await self._aclient.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=self.batch_size,
            parallel=self.parallel,
            max_retries=self.max_retries,
            wait=True,
        )

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        from qdrant_client.http import models as rest

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=ref_doc_id)
                    )
                ]
            ),
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Asynchronous method to delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        from qdrant_client.http import models as rest

        await self._aclient.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=ref_doc_id)
                    )
                ]
            ),
        )

    @property
    def client(self) -> Any:
        """Return the Qdrant client."""
        return self._client

    def _create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a Qdrant collection."""
        from qdrant_client.http import models as rest
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            if self.enable_hybrid:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "text-dense": rest.VectorParams(
                            size=vector_size,
                            distance=rest.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "text-sparse": rest.SparseVectorParams(
                            index=rest.SparseIndexParams()
                        )
                    },
                )
            else:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(
                        size=vector_size,
                        distance=rest.Distance.COSINE,
                    ),
                )
        except (ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            logger.warning(
                "Collection %s already exists, skipping collection creation.",
                collection_name,
            )
        self._collection_initialized = True

    async def _acreate_collection(self, collection_name: str, vector_size: int) -> None:
        """Asynchronous method to create a Qdrant collection."""
        from qdrant_client.http import models as rest
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            if self.enable_hybrid:
                await self._aclient.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "text-dense": rest.VectorParams(
                            size=vector_size,
                            distance=rest.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "text-sparse": rest.SparseVectorParams(
                            index=rest.SparseIndexParams()
                        )
                    },
                )
            else:
                await self._aclient.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(
                        size=vector_size,
                        distance=rest.Distance.COSINE,
                    ),
                )
        except (ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            logger.warning(
                "Collection %s already exists, skipping collection creation.",
                collection_name,
            )
        self._collection_initialized = True

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        from grpc import RpcError
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            self._client.get_collection(collection_name)
        except (RpcError, UnexpectedResponse, ValueError):
            return False
        return True

    async def _acollection_exists(self, collection_name: str) -> bool:
        """Asynchronous method to check if a collection exists."""
        from grpc import RpcError
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            await self._aclient.get_collection(collection_name)
        except (RpcError, UnexpectedResponse, ValueError):
            return False
        return True

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
        from qdrant_client import models as rest
        from qdrant_client.http.models import Filter

        query_embedding = cast(List[float], query.query_embedding)
        #  NOTE: users can pass in qdrant_filters (nested/complicated filters) to override the default MetadataFilters
        qdrant_filters = kwargs.get("qdrant_filters")
        if qdrant_filters is not None:
            query_filter = qdrant_filters
        else:
            query_filter = cast(Filter, self._build_query_filter(query))

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
                            name="text-dense",
                            vector=query_embedding,
                        ),
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=True,
                    ),
                    rest.SearchRequest(
                        vector=rest.NamedSparseVector(
                            name="text-sparse",
                            vector=rest.SparseVector(
                                indices=sparse_indices[0],
                                values=sparse_embedding[0],
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
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
                            name="text-sparse",
                            vector=rest.SparseVector(
                                indices=sparse_indices[0],
                                values=sparse_embedding[0],
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
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
                            name="text-dense",
                            vector=query_embedding,
                        ),
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=True,
                    ),
                ],
            )

            return self.parse_to_query_result(response[0])
        else:
            response = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=query.similarity_top_k,
                query_filter=query_filter,
            )
            return self.parse_to_query_result(response)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Asynchronous method to query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query
        """
        from qdrant_client import models as rest
        from qdrant_client.http.models import Filter

        query_embedding = cast(List[float], query.query_embedding)

        #  NOTE: users can pass in qdrant_filters (nested/complicated filters) to override the default MetadataFilters
        qdrant_filters = kwargs.get("qdrant_filters")
        if qdrant_filters is not None:
            query_filter = qdrant_filters
        else:
            # build metadata filters
            query_filter = cast(Filter, self._build_query_filter(query))

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
                            name="text-dense",
                            vector=query_embedding,
                        ),
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=True,
                    ),
                    rest.SearchRequest(
                        vector=rest.NamedSparseVector(
                            name="text-sparse",
                            vector=rest.SparseVector(
                                indices=sparse_indices[0],
                                values=sparse_embedding[0],
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
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
                            name="text-sparse",
                            vector=rest.SparseVector(
                                indices=sparse_indices[0],
                                values=sparse_embedding[0],
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
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
                            name="text-dense",
                            vector=query_embedding,
                        ),
                        limit=query.similarity_top_k,
                        filter=query_filter,
                        with_payload=True,
                    ),
                ],
            )

            return self.parse_to_query_result(response[0])
        else:
            response = await self._aclient.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=query.similarity_top_k,
                query_filter=query_filter,
            )

            return self.parse_to_query_result(response)

    def parse_to_query_result(self, response: List[Any]) -> VectorStoreQueryResult:
        """
        Convert vector store response to VectorStoreQueryResult.

        Args:
            response: List[Any]: List of results returned from the vector store.
        """
        from qdrant_client.http.models import Payload

        nodes = []
        similarities = []
        ids = []

        for point in response:
            payload = cast(Payload, point.payload)
            try:
                node = metadata_dict_to_node(payload)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                logger.debug("Failed to parse Node metadata, fallback to legacy logic.")
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    payload
                )

                node = TextNode(
                    id_=str(point.id),
                    text=payload.get("text"),
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )
            nodes.append(node)
            similarities.append(point.score)
            ids.append(str(point.id))

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _build_query_filter(self, query: VectorStoreQuery) -> Optional[Any]:
        if not query.doc_ids and not query.query_str:
            return None

        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchExcept,
            MatchText,
            MatchValue,
            Range,
        )

        must_conditions = []

        if query.doc_ids:
            must_conditions.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=query.doc_ids),
                )
            )

        if query.node_ids:
            must_conditions.append(
                FieldCondition(
                    key="id",
                    match=MatchAny(any=query.node_ids),
                )
            )

        # Qdrant does not use the query.query_str property for the filtering. Full-text
        # filtering cannot handle longer queries and can effectively filter our all the
        # nodes. See: https://github.com/jerryjliu/llama_index/pull/1181

        if query.filters is None:
            return Filter(must=must_conditions)

        for subfilter in query.filters.filters:
            # only for exact match
            if not subfilter.operator or subfilter.operator == "==":
                if isinstance(subfilter.value, float):
                    must_conditions.append(
                        FieldCondition(
                            key=subfilter.key,
                            range=Range(
                                gte=subfilter.value,
                                lte=subfilter.value,
                            ),
                        )
                    )
                else:
                    must_conditions.append(
                        FieldCondition(
                            key=subfilter.key,
                            match=MatchValue(value=subfilter.value),
                        )
                    )
            elif subfilter.operator == "<":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(lt=subfilter.value),
                    )
                )
            elif subfilter.operator == ">":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(gt=subfilter.value),
                    )
                )
            elif subfilter.operator == ">=":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(gte=subfilter.value),
                    )
                )
            elif subfilter.operator == "<=":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(lte=subfilter.value),
                    )
                )
            elif subfilter.operator == "text_match":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchText(text=subfilter.value),
                    )
                )
            elif subfilter.operator == "!=":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchExcept(**{"except": [subfilter.value]}),
                    )
                )

        return Filter(must=must_conditions)
