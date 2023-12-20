"""Qdrant vector store index.

An index that is built on top of an existing Qdrant collection.

"""
import logging
from typing import Any, Callable, List, Optional, cast

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.utils import iter_batch
from llama_index.vector_stores.qdrant_utils import (
    SparseEncoderCallable,
    default_sparse_encoder,
)
from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)
import_err_msg = (
    "`qdrant-client` package not found, please run `pip install qdrant-client`"
)


class QdrantVectorStore(BasePydanticVectorStore):
    """Qdrant Vector Store.

    In this vector store, embeddings and docs are stored within a
    Qdrant collection.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes.

    Args:
        collection_name: (str): name of the Qdrant collection
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
    """

    stores_text: bool = True
    flat_metadata: bool = False

    collection_name: str
    url: Optional[str]
    api_key: Optional[str]
    batch_size: int
    prefer_grpc: bool
    client_kwargs: dict = Field(default_factory=dict)
    enable_hybrid: bool

    _client: Any = PrivateAttr()
    _collection_initialized: bool = PrivateAttr()
    _sparse_doc_fn: Optional[SparseEncoderCallable] = PrivateAttr()
    _sparse_query_fn: Optional[SparseEncoderCallable] = PrivateAttr()
    _hybrid_fusion_fn: Optional[
        Callable[
            [VectorStoreQueryResult, VectorStoreQueryResult], VectorStoreQueryResult
        ]
    ] = PrivateAttr()

    def __init__(
        self,
        collection_name: str,
        client: Optional[Any] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        prefer_grpc: bool = False,
        client_kwargs: Optional[dict] = None,
        enable_hybrid: bool = False,
        sparse_doc_fn: Optional[SparseEncoderCallable] = None,
        sparse_query_fn: Optional[SparseEncoderCallable] = None,
        hybrid_fusion_fn: Optional[
            Callable[
                [VectorStoreQueryResult, VectorStoreQueryResult], VectorStoreQueryResult
            ]
        ] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            import qdrant_client
        except ImportError:
            raise ImportError(import_err_msg)

        if client is None and (
            url is None or api_key is None or collection_name is None
        ):
            raise ValueError(
                "Must provide either a QdrantClient instance or a url and api_key."
            )

        if client is None:
            client_kwargs = client_kwargs or {}
            self._client = qdrant_client.QdrantClient(
                url=url, api_key=api_key, **client_kwargs
            )
        else:
            self._client = cast(qdrant_client.QdrantClient, client)

        self._collection_initialized = self._collection_exists(collection_name)

        # setup hybrid search if enabled
        if enable_hybrid:
            self._sparse_doc_fn = sparse_doc_fn or default_sparse_encoder(
                "naver/efficient-splade-VI-BT-large-doc"
            )
            self._sparse_query_fn = sparse_query_fn or default_sparse_encoder(
                "naver/efficient-splade-VI-BT-large-doc"
            )
        self._hybrid_fusion_fn = hybrid_fusion_fn or self.deduplicate_hybrid_result

        super().__init__(
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            batch_size=batch_size,
            prefer_grpc=prefer_grpc,
            client_kwargs=client_kwargs or {},
            enable_hybrid=enable_hybrid,
        )

    @classmethod
    def class_name(cls) -> str:
        return "QdrantVectorStore"

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        from qdrant_client.http import models as rest

        if len(nodes) > 0 and not self._collection_initialized:
            self._create_collection(
                collection_name=self.collection_name,
                vector_size=len(nodes[0].get_embedding()),
            )

        ids = []
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
                    vectors.append(node.get_embedding())

                metadata = node_to_metadata_dict(
                    node, remove_text=False, flat_metadata=self.flat_metadata
                )

                payloads.append(metadata)

            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(id=node_id, payload=payload, vector=vector)
                    for node_id, payload, vector in zip(node_ids, payloads, vectors)
                ],
            )
            ids.extend(node_ids)
        return ids

    async def async_add(self, nodes: List[BaseNode]) -> List[str]:
        """Asynchronous method to add nodes to Qdrant index.

        Args:
            nodes: List[BaseNode]: List of nodes with embeddings.

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ValueError: If trying to using async methods without
                            setting `prefer_grpc` to True.
        """
        if not self.prefer_grpc:
            raise ValueError(
                "`prefer_grpc` must be set to True to use async insertion."
            )

        from qdrant_client import grpc

        if len(nodes) > 0 and not self._collection_initialized:
            await self._async_create_collection(
                collection_name=self.collection_name,
                vector_size=len(nodes[0].get_embedding()),
            )

        ids = []
        for node_batch in iter_batch(nodes, self.batch_size):
            node_ids = []
            grpc_points = []
            for node in node_batch:
                assert isinstance(node, BaseNode)
                node_ids.append(node.node_id)
                grpc_points.append(
                    grpc.PointStruct(
                        id=grpc.PointId(num=node.node_id),
                        payload=self._get_async_payload(
                            node_to_metadata_dict(
                                node,
                                remove_text=False,
                                flat_metadata=self.flat_metadata,
                            )
                        ),
                        vectors=grpc.Vectors(
                            vector=grpc.Vector(data=node.get_embedding()),
                        ),
                    )
                )

            await self._client.async_grpc_points.Upsert(
                grpc.UpsertPoints(
                    collection_name=self.collection_name,
                    points=grpc_points,
                )
            )
            ids.extend(node_ids)
        return ids

    def _get_async_payload(self, metadata: dict) -> dict:
        """Convert the metadata payload to formatted Qdrant payload.

        Args:
            metadata (dict): Metadata of a node.

        """
        grpc_payload = {}

        for key, value in metadata.items():
            grpc_payload[key] = self._value_to_grpc_value(value)

        return grpc_payload

    def _value_to_grpc_value(self, value: Any) -> Optional[Any]:
        """Convert the REST value to gRPC value.

        Raises:
            ValueError: If an unsupported value is passed.
        """
        from qdrant_client.grpc import ListValue, NullValue, Struct, Value

        if value is None:
            return Value(null_value=NullValue.NULL_VALUE)
        if isinstance(value, bool):
            return Value(bool_value=value)
        if isinstance(value, int):
            return Value(integer_value=value)
        if isinstance(value, float):
            return Value(double_value=value)
        if isinstance(value, str):
            return Value(string_value=value)
        if isinstance(value, list):
            return Value(
                list_value=ListValue(
                    values=[self._value_to_grpc_value(v) for v in value]
                )
            )
        if isinstance(value, dict):
            return Value(
                struct_value=Struct(
                    fields={k: self._value_to_grpc_value(v) for k, v in value.items()}
                )
            )

        raise ValueError(f"{value} is not a supported value.")

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
        from qdrant_client import grpc

        await self._client.async_grpc_points.Delete(
            grpc.DeletePoints(
                collection_name=self.collection_name,
                points=grpc.PointsSelector(
                    filter=grpc.Filter(
                        must=[
                            grpc.Condition(
                                field=grpc.FieldCondition(
                                    key="doc_id", match=grpc.Match(text=ref_doc_id)
                                )
                            )
                        ]
                    )
                ),
            )
        )

    @property
    def client(self) -> Any:
        """Return the Qdrant client."""
        return self._client

    def _create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a Qdrant collection."""
        from qdrant_client.http import models as rest

        if self.enable_hybrid:
            self._client.recreate_collection(
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
            self._client.recreate_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )
        self._collection_initialized = True

    async def _async_create_collection(
        self, collection_name: str, vector_size: int
    ) -> None:
        """Asynchronous method to create a Qdrant collection."""
        from qdrant_client import grpc

        await self._client.async_grpc_collections.Create(
            grpc.CreateCollection(
                collection_name=collection_name,
                vectors_config=grpc.VectorsConfig(
                    params=grpc.VectorParams(
                        size=vector_size,
                        distance=grpc.Distance.Cosine,
                    )
                ),
            )
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

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query
        """
        from qdrant_client import models as rest
        from qdrant_client.http.models import Filter

        query_embedding = cast(List[float], query.query_embedding)
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
            )
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
        """Asynchronous method to query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query
        """
        from qdrant_client import grpc
        from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc
        from qdrant_client.http.models import Filter

        query_embedding = cast(List[float], query.query_embedding)
        query_filter = RestToGrpc.convert_filter(
            cast(Filter, self._build_query_filter(query))
        )

        res = await self._client.async_grpc_points.Search(
            grpc.SearchPoints(
                collection_name=self.collection_name,
                vector=query_embedding,
                filter=query_filter,
                limit=cast(int, query.similarity_top_k),
                with_payload=grpc.WithPayloadSelector(enable=True),
            )
        )

        response = [GrpcToRest.convert_scored_point(hit) for hit in res.result]

        return self.parse_to_query_result(response=response)

    def parse_to_query_result(self, response: List[Any]) -> VectorStoreQueryResult:
        """Convert vector store response to VectorStoreQueryResult.

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

    def deduplicate_hybrid_result(
        self,
        dense_result: VectorStoreQueryResult,
        sparse_reuslt: VectorStoreQueryResult,
    ) -> VectorStoreQueryResult:
        """Deduplicate hybrid results."""
        ids = [*(dense_result.ids or []), *(sparse_reuslt.ids or [])]
        similarities = [
            *(dense_result.similarities or []),
            *(sparse_reuslt.similarities or []),
        ]
        nodes = [*(dense_result.nodes or []), *(sparse_reuslt.nodes or [])]

        dedup_ids = []
        dedup_similarities = []
        dedup_nodes = []
        for id_ in ids:
            if id_ not in dedup_ids:
                dedup_ids.append(id_)
                dedup_similarities.append(similarities[ids.index(id_)])
                dedup_nodes.append(nodes[ids.index(id_)])

        return VectorStoreQueryResult(
            ids=dedup_ids,
            similarities=dedup_similarities,
            nodes=dedup_nodes,
        )

    def _build_query_filter(self, query: VectorStoreQuery) -> Optional[Any]:
        if not query.doc_ids and not query.query_str:
            return None

        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchAny,
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

        return Filter(must=must_conditions)
