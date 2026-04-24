"""Actian Vector AI vector store index."""

from typing import Any, Dict, List, Optional, Sequence

from typing_extensions import Self

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStoreQueryMode,
)

from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from actian_vectorai import (
    Condition,
    Field,
    FilterBuilder,
    VectorAIClient,
    AsyncVectorAIClient,
    Filter,
    is_empty,
    has_id,
)

from actian_vectorai.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    UpdateStatus,
    VectorParams,
    RetrievedPoint,
)


class ActianVectorAIVectorStore(BasePydanticVectorStore):
    """
    Vector store backend backed by an Actian Vector AI collection.

    """

    stores_text: bool = False
    flat_metadata: bool = False

    url: str
    collection_name: str
    client_kwargs: Optional[Dict[str, Any]]
    collection_kwargs: Optional[Dict[str, Any]]
    dense_vector_name: str
    dense_vector_params: Optional[VectorParams]

    _client: VectorAIClient = PrivateAttr(None)
    _async_client: AsyncVectorAIClient = PrivateAttr(None)
    _collection_exists: bool = PrivateAttr(False)
    _clear_existing_collection: bool = PrivateAttr(False)
    _lazy_collection_exist_check: bool = PrivateAttr(False)

    def __init__(
        self,
        url: str = "localhost:6574",
        collection_name: str = "llama_index_collection",
        client_kwargs: Optional[dict[str, Any]] = None,
        dense_vector_name: str = "llama_index_dense_vector",
        dense_vector_params: Optional[VectorParams] = None,
        collection_kwargs: Optional[dict[str, Any]] = None,
        stores_text: bool = False,
        clear_existing_collection: bool = False,
        client: Optional[VectorAIClient] = None,
        async_client: Optional[AsyncVectorAIClient] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an Actian Vector AI-backed vector store.

        Args:
            url: Actian Vector AI endpoint in host:port format. Ignored when
                explicit sync or async clients are provided. Defaults to
                "localhost:6574".
            collection_name: Name of the collection used to store vectors and
                metadata. Defaults to "llama_index_collection".
            client_kwargs: Optional keyword arguments forwarded to internally
                created sync/async clients when explicit client instances are
                not provided. Ignored if client or async_client is supplied.
            dense_vector_name: Dense vector name in the collection. Defaults to
                "llama_index_dense_vector".
            dense_vector_params: Optional vector configuration for the dense
                vector field (size, distance metric). If omitted, defaults are
                inferred from the first inserted embedding (inferred size) and
                cosine distance.
            collection_kwargs: Optional keyword arguments passed to collection
                creation. The vectors_config will be replaced by the dense_vector_name
                and dense_vector_params arguments and should not be included here.
            stores_text: Whether node text should be stored in payload metadata.
                Defaults to False.
            clear_existing_collection: Whether to delete an existing collection
                with the same name before the first operation. Defaults to False.
            client: Optional pre-configured synchronous VectorAIClient instance.
                If provided, url and client_kwargs are ignored. Mutually
                exclusive with async_client only.
            async_client: Optional pre-configured asynchronous AsyncVectorAIClient
                instance. If provided, url and client_kwargs are ignored. Must not
                be the same instance as the internal async client of a provided
                synchronous client.
            **kwargs: Additional arguments forwarded to BasePydanticVectorStore.

        Raises:
            ValueError: If client is not a VectorAIClient instance, if async_client
                is not an AsyncVectorAIClient instance, or if async_client is the
                same instance used internally by the provided synchronous client.

        """
        super().__init__(
            url=url,
            collection_name=collection_name,
            client_kwargs=client_kwargs,
            collection_kwargs=collection_kwargs,
            stores_text=stores_text,
            dense_vector_name=dense_vector_name,
            dense_vector_params=dense_vector_params,
            **kwargs,
        )

        self._clear_existing_collection = clear_existing_collection

        if client is None and async_client is None:
            self._client = VectorAIClient(url=url, **(client_kwargs or {}))
            self._async_client = AsyncVectorAIClient(url=url, **(client_kwargs or {}))
        else:
            if client is not None:
                if not isinstance(client, VectorAIClient):
                    raise ValueError("client must be an instance of VectorAIClient.")
                self._client = client
                if client._async_client is async_client:
                    raise ValueError(
                        "async_client cannot be the same instance as the async client used by the provided synchronous client. Please provide a different AsyncVectorAIClient instance if you wish to provide an async client."
                    )
            if async_client is not None:
                if not isinstance(async_client, AsyncVectorAIClient):
                    raise ValueError(
                        "async_client must be an instance of AsyncVectorAIClient."
                    )
                self._async_client = async_client

    def __enter__(self) -> Self:
        """
        Enter synchronous context and ensure the client is connected.

        Returns:
            This vector store instance.

        Raises:
            ValueError: If the synchronous client is not initialized.

        """
        if self._client is None:
            raise ValueError(
                "Synchronous client is not initialized. Please initialize the ActianVectorAIVectorStore with a VectorAIClient to use synchronous methods."
            )
        if not self._client.is_connected:
            self._client.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit synchronous context and close the client connection.

        Args:
            exc_type: Exception class raised in the context, if any.
            exc_val: Exception instance raised in the context, if any.
            exc_tb: Traceback associated with the raised exception, if any.

        """
        if self._client.is_connected:
            self._client.shutdown()

    async def __aenter__(self) -> Self:
        """
        Enter asynchronous context and ensure the async client is connected.

        Returns:
            This vector store instance.

        Raises:
            ValueError: If the asynchronous client is not initialized.

        """
        if self._async_client is None:
            raise ValueError(
                "Async client is not initialized. Please initialize the ActianVectorAIVectorStore with an AsyncVectorAIClient to use asynchronous methods."
            )
        if not self._async_client.is_connected:
            await self._async_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit asynchronous context and close the async client connection.

        Args:
            exc_type: Exception class raised in the context, if any.
            exc_val: Exception instance raised in the context, if any.
            exc_tb: Traceback associated with the raised exception, if any.

        """
        if self._async_client is not None and self._async_client.is_connected:
            await self._async_client.close()

    @classmethod
    def class_name(cls) -> str:
        return "ActianVectorAIVectorStore"

    @property
    def client(self) -> Any:
        """Return Actian Vector AI client."""
        if self._client is None:
            raise ValueError(
                "Synchronous client is not initialized. Please initialize the ActianVectorAIVectorStore with a VectorAIClient to use synchronous methods."
            )
        return self._client

    @property
    def async_client(self) -> Any:
        """Return Actian Vector AI async client."""
        if self._async_client is None:
            raise ValueError(
                "Async client is not initialized. Please initialize the ActianVectorAIVectorStore with an AsyncVectorAIClient to use asynchronous methods."
            )
        return self._async_client

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        limit: Optional[int] = 9999,
    ) -> List[BaseNode]:
        """
        Fetch nodes by id and/or metadata filters.

        Args:
            node_ids: Optional list of node IDs to match.
            filters: Optional metadata filters to apply.
            limit: Optional maximum number of nodes to return. Defaults to 9999.

        Returns:
            Matching nodes in the collection.

        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return []

        result = self._client.points.scroll(
            self.collection_name,
            filter=self._build_db_filter_from_node_ids_doc_ids_and_metadata_filters(
                node_ids=node_ids, doc_ids=None, filters=filters
            ),
            limit=limit,
        )

        return self._build_base_nodes_from_retrieved_points(result[0])

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        limit: Optional[int] = 9999,
    ) -> List[BaseNode]:
        """
        Async version of get_nodes.

        Args:
            node_ids: Optional list of node IDs to match.
            filters: Optional metadata filters to apply.
            limit: Optional maximum number of nodes to return. Defaults to 9999.

        Returns:
            Matching nodes in the collection.

        """
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return []

        result = await self._async_client.points.scroll(
            self.collection_name,
            filter=self._build_db_filter_from_node_ids_doc_ids_and_metadata_filters(
                node_ids=node_ids, doc_ids=None, filters=filters
            ),
            limit=limit,
        )

        return self._build_base_nodes_from_retrieved_points(result[0])

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Insert nodes into the collection.

        Args:
            nodes: Nodes with embeddings to upsert.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility. Currently unused.

        Returns:
            IDs of inserted/updated nodes.

        """
        self._lazy_client_operation_check()

        if not len(nodes) > 0:
            return []

        self._create_collection_if_not_exists(len(nodes[0].get_embedding()))

        points, ids = self._build_points_from_nodes(nodes)
        result = self._client.points.upsert(self.collection_name, points)
        assert result.status == UpdateStatus.Completed, (
            f"Failed to add points to collection {self.collection_name}. Response: {result}"
        )
        return ids

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Async version of add.

        Args:
            nodes: Nodes with embeddings to upsert.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility. Currently unused.

        Returns:
            IDs of inserted/updated nodes.

        """
        await self._lazy_async_client_operation_check()

        if not len(nodes) > 0:
            return []

        await self._acreate_collection_if_not_exists(len(nodes[0].get_embedding()))

        points, ids = self._build_points_from_nodes(nodes)
        result = await self._async_client.points.upsert(self.collection_name, points)
        assert result.status == UpdateStatus.Completed, (
            f"Failed to add points to collection {self.collection_name}. Response: {result}"
        )
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete all nodes associated with a document ID.

        Args:
            ref_doc_id: Reference document ID to match.
            **delete_kwargs: Additional keyword arguments accepted for interface
                compatibility. Currently unused.

        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return

        result = self._client.points.delete(
            self.collection_name,
            filter=FilterBuilder().must(Field("ref_doc_id").eq(ref_doc_id)).build(),
        )

        assert result.status == UpdateStatus.Completed, (
            f"Failed to delete points with ref_doc_id {ref_doc_id} from collection {self.collection_name}. Response: {result}"
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Async version of delete by reference document ID.

        Args:
            ref_doc_id: Reference document ID to match.
            **delete_kwargs: Additional keyword arguments accepted for interface
                compatibility. Currently unused.

        """
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return

        result = await self._async_client.points.delete(
            self.collection_name,
            filter=FilterBuilder().must(Field("ref_doc_id").eq(ref_doc_id)).build(),
        )

        assert result.status == UpdateStatus.Completed, (
            f"Failed to delete points with ref_doc_id {ref_doc_id} from collection {self.collection_name}. Response: {result}"
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes by IDs and/or metadata filters.

        Args:
            node_ids: Optional list of node IDs to delete.
            filters: Optional metadata filters to constrain deletion.
            **delete_kwargs: Additional keyword arguments accepted for interface
                compatibility. Currently unused.

        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return

        result = self._client.points.delete(
            self.collection_name,
            ids=node_ids,
            filter=self._build_db_filter_from_metadata_filters(filters),
        )

        assert result.status == UpdateStatus.Completed, (
            f"Failed to delete points from collection {self.collection_name}. Response: {result}"
        )

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Async version of delete_nodes.

        Args:
            node_ids: Optional list of node IDs to delete.
            filters: Optional metadata filters to constrain deletion.
            **delete_kwargs: Additional keyword arguments accepted for interface
                compatibility. Currently unused.

        """
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return

        result = await self._async_client.points.delete(
            self.collection_name,
            ids=node_ids,
            filter=self._build_db_filter_from_metadata_filters(filters),
        )

        assert result.status == UpdateStatus.Completed, (
            f"Failed to delete points with ids {node_ids} from collection {self.collection_name}. Response: {result}"
        )

    def clear(self) -> None:
        """Delete the entire collection and reset local existence state."""
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return

        result = self._client.collections.delete(self.collection_name)
        assert result, (
            f"Failed to clear collection {self.collection_name}. Response: {result}"
        )

        self._collection_exists = False

    async def aclear(self) -> None:
        """Async version of clear collection state and data."""
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return

        result = await self._async_client.collections.delete(self.collection_name)
        assert result, (
            f"Failed to clear collection {self.collection_name}. Response: {result}"
        )

        self._collection_exists = False

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Run similarity search and return matching nodes.

        Args:
            query: Query payload containing embedding, filters, and top-k.
            **kwargs: Additional keyword arguments forwarded to the underlying
                Actian search API.

        Returns:
            Search result containing nodes, scores, and IDs.

        Note:
            Only VectorStoreQueryMode.DEFAULT is currently supported.

        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        if query.mode == VectorStoreQueryMode.DEFAULT:
            results = self.client.points.search(
                self.collection_name,
                query.query_embedding,
                using=self.dense_vector_name,
                limit=query.similarity_top_k,
                filter=self._build_db_filter_from_vector_store_query(query),
                **kwargs,
            )
            return self._build_vector_store_query_result_from_scored_points(results)

        raise NotImplementedError(
            "Only DEFAULT query mode is supported for ActianVectorAIVectorStore."
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Async version of query.

        Args:
            query: Query payload containing embedding, filters, and top-k.
            **kwargs: Additional keyword arguments forwarded to the underlying
                Actian search API.

        Returns:
            Search result containing nodes, scores, and IDs.

        Note:
            Only VectorStoreQueryMode.DEFAULT is currently supported.

        """
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        if query.mode == VectorStoreQueryMode.DEFAULT:
            results = await self._async_client.points.search(
                self.collection_name,
                query.query_embedding,
                using=self.dense_vector_name,
                limit=query.similarity_top_k,
                filter=self._build_db_filter_from_vector_store_query(query),
                **kwargs,
            )
            return self._build_vector_store_query_result_from_scored_points(results)

        raise NotImplementedError(
            "Only DEFAULT query mode is supported for ActianVectorAIVectorStore."
        )

    def connect(self) -> None:
        """Connect to Actian Vector AI client."""
        if self._client is None:
            raise ValueError(
                "Synchronous client is not initialized. Please initialize the ActianVectorAIVectorStore with a VectorAIClient to use synchronous methods."
            )
        if not self._client.is_connected:
            self._client.connect()

    async def aconnect(self) -> None:
        """Asynchronously connect to Actian Vector AI client."""
        if self._async_client is None:
            raise ValueError(
                "Async client is not initialized. Please initialize the ActianVectorAIVectorStore with an AsyncVectorAIClient to use asynchronous methods."
            )
        if not self._async_client.is_connected:
            await self._async_client.connect()

    def close(self) -> None:
        """Close connection to Actian Vector AI client."""
        if self._client is not None and self._client.is_connected:
            self._client.shutdown()

    async def aclose(self) -> None:
        """Asynchronously close connection to Actian Vector AI client."""
        if self._async_client is not None and self._async_client.is_connected:
            await self._async_client.close()

    def _build_points_from_nodes(
        self, nodes: List[BaseNode]
    ) -> tuple[List[PointStruct], List[str]]:
        """
        Convert nodes to Actian Vector AI PointStruct objects.

        Args:
            nodes: Nodes with embeddings and metadata.

        Returns:
            Tuple of points for upsert and their corresponding node IDs.

        """
        points = []
        ids = []

        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=not self.stores_text, flat_metadata=self.flat_metadata
            )

            _vector = {}

            _vector[self.dense_vector_name] = node.get_embedding()

            point = PointStruct(
                id=node.node_id,
                vector=_vector,
                payload=metadata,
            )

            points.append(point)
            ids.append(node.node_id)
        return points, ids

    def _build_db_filter_from_metadata_filters(
        self, filters: MetadataFilters
    ) -> Filter:
        """
        Translate LlamaIndex metadata filters to an Actian filter.

        Args:
            filters: Nested metadata filter expression.

        Returns:
            Equivalent Actian filter object, or None when filters is None.

        """
        if filters is None:
            return None

        conditions = []
        for filter in filters.filters:
            if isinstance(filter, MetadataFilters):
                if len(filter.filters) == 0:
                    continue
                conditions.append(
                    Condition(
                        filter=self._build_db_filter_from_metadata_filters(filter)
                    )
                )
            else:

                def filter_operation_to_condition_eq(key: str, value: Any) -> Condition:
                    if isinstance(value, float):
                        return Field(key).between(value, value)
                    else:
                        return Field(key).eq(value)

                def filter_operation_to_condition_ne(key: str, value: Any) -> Condition:
                    if isinstance(value, float):
                        return Condition(
                            filter=FilterBuilder()
                            .should(Field(key).lt(value))
                            .should(Field(key).gt(value))
                            .build()
                        )
                    else:
                        return Field(key).except_of([value])

                def filter_operation_to_condition_in(key: str, value: Any) -> Condition:
                    if isinstance(value, list):
                        values = value
                    else:
                        values = value.split(",")
                    return Field(key).any_of(values)

                def filter_operation_to_condition_nin(
                    key: str, value: Any
                ) -> Condition:
                    if isinstance(value, list):
                        values = value
                    else:
                        values = value.split(",")
                    return Field(key).except_of(values)

                fops_dict = {
                    FilterOperator.EQ: filter_operation_to_condition_eq,
                    FilterOperator.GT: lambda key, value: Field(key).gt(float(value)),
                    FilterOperator.LT: lambda key, value: Field(key).lt(float(value)),
                    FilterOperator.NE: filter_operation_to_condition_ne,
                    FilterOperator.GTE: lambda key, value: Field(key).gte(float(value)),
                    FilterOperator.LTE: lambda key, value: Field(key).lte(float(value)),
                    FilterOperator.IN: filter_operation_to_condition_in,
                    FilterOperator.NIN: filter_operation_to_condition_nin,
                    FilterOperator.TEXT_MATCH: lambda key, value: Field(key).text(
                        value
                    ),
                    FilterOperator.IS_EMPTY: lambda key, value: is_empty(key),
                }

                if filter.operator not in fops_dict:
                    raise NotImplementedError(
                        f"Unsupported filter operator: {filter.operator}"
                    )
                conditions.append(fops_dict[filter.operator](filter.key, filter.value))

        if filters.condition == FilterCondition.AND:
            return Filter(must=conditions)
        elif filters.condition == FilterCondition.OR:
            return Filter(should=conditions)
        elif filters.condition == FilterCondition.NOT:
            return Filter(must_not=conditions)
        else:
            raise NotImplementedError(
                f"Unsupported filter condition: {filters.condition}"
            )

    def _build_db_filter_from_vector_store_query(
        self, query: VectorStoreQuery
    ) -> Filter:
        """
        Build an Actian filter from query-level constraints.

        Args:
            query: Vector store query with optional node/doc IDs and metadata filters.

        Returns:
            Actian filter that combines all supplied constraints.

        """
        return self._build_db_filter_from_node_ids_doc_ids_and_metadata_filters(
            node_ids=query.node_ids,
            doc_ids=query.doc_ids,
            filters=query.filters,
        )

    def _build_db_filter_from_node_ids_doc_ids_and_metadata_filters(
        self,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> Filter:
        """
        Build an Actian filter from node IDs, document IDs, and metadata filters.

        Args:
            node_ids: Optional list of node IDs to match.
            doc_ids: Optional list of document IDs to match.
            filters: Optional metadata filters to apply.

        Returns:
            Actian filter that combines all supplied constraints.

        """
        conditions = []

        if node_ids is not None and len(node_ids) > 0:
            conditions.append(has_id(node_ids))

        if doc_ids is not None and len(doc_ids) > 0:
            conditions.append(Field("ref_doc_id").any_of(doc_ids))

        if filters is not None:
            conditions.append(
                Condition(filter=self._build_db_filter_from_metadata_filters(filters))
            )

        return Filter(must=conditions) if conditions else None

    def _build_vector_store_query_result_from_scored_points(
        self, scored_points: List[ScoredPoint]
    ) -> VectorStoreQueryResult:
        """
        Convert scored Actian search points into LlamaIndex query results.

        Args:
            scored_points: Search hits returned by Actian Vector AI.

        Returns:
            LlamaIndex query result containing nodes, similarity scores, and IDs.

        """
        ids = []
        nodes = []
        similarities = []
        for point in scored_points:
            id = point.id
            node = metadata_dict_to_node(point.payload)
            node.embedding = point.vectors
            similarity = point.score

            ids.append(id)
            nodes.append(node)
            similarities.append(similarity)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _build_base_nodes_from_retrieved_points(
        self, retrieved_points: List[RetrievedPoint]
    ) -> List[BaseNode]:
        """
        Convert Actian RetrievedPoint objects into LlamaIndex BaseNode objects.

        Args:
            retrieved_points: List of RetrievedPoint objects returned by Actian Vector AI.

        Returns:
            List of BaseNode objects constructed from the retrieved points.

        """
        nodes = []
        for point in retrieved_points:
            node = metadata_dict_to_node(point.payload)
            node.embedding = point.vectors
            nodes.append(node)

        return nodes

    def _lazy_client_operation_check(self) -> None:
        """
        Validate synchronous client state and lazily initialize collection flags.

        Raises:
            ValueError: If the synchronous client is not initialized.
            ConnectionError: If the synchronous client is not connected.

        """
        if self._client is None:
            raise ValueError(
                "Synchronous client is not initialized. Please initialize the ActianVectorAIVectorStore with a VectorAIClient to use synchronous methods."
            )
        if not self._client.is_connected:
            raise ConnectionError(
                "Synchronous client is not connected. Please ensure the client is connected before using synchronous methods. Please use with statement to automatically connect and disconnect the client. Alternatively, you can manually call the connect() method to connect the client before performing operations and close() method to disconnect the client after performing operations."
            )

        if self._clear_existing_collection:
            if self._client.collections.exists(self.collection_name):
                self._client.collections.delete(self.collection_name)
            self._clear_existing_collection = False
            self._collection_exists = False

        if not self._lazy_collection_exist_check:
            if self._client.collections.exists(self.collection_name):
                self._collection_exists = True
            self._lazy_collection_exist_check = True

    async def _lazy_async_client_operation_check(self) -> None:
        """
        Validate async client state and lazily initialize collection flags.

        Raises:
            ValueError: If the async client is not initialized.
            ConnectionError: If the async client is not connected.

        """
        if self._async_client is None:
            raise ValueError(
                "Async client is not initialized. Please initialize the ActianVectorAIVectorStore with an AsyncVectorAIClient to use asynchronous methods."
            )
        if not self._async_client.is_connected:
            raise ConnectionError(
                "Async client is not connected. Please ensure the async client is connected before using asynchronous methods. Please use async with statement to automatically connect and disconnect the async client. Alternatively, you can manually call the aconnect() method to connect the async client before performing operations and aclose() method to disconnect the async client after performing operations."
            )

        if self._clear_existing_collection:
            if await self._async_client.collections.exists(self.collection_name):
                await self._async_client.collections.delete(self.collection_name)
            self._clear_existing_collection = False
            self._collection_exists = False

        if not self._lazy_collection_exist_check:
            if await self._async_client.collections.exists(self.collection_name):
                self._collection_exists = True
            self._lazy_collection_exist_check = True

    def _create_collection_if_not_exists(self, embed_dim: int) -> None:
        """
        Create collection when missing using the configured vector schema.

        Args:
            embed_dim: Embedding dimensionality used to initialize default vector
                parameters when not explicitly configured.

        """
        if not self._collection_exists:
            _collection_kwargs = self._prepare_collection_kwargs(embed_dim)
            self._client.collections.create(self.collection_name, **_collection_kwargs)
            self._collection_exists = True

    async def _acreate_collection_if_not_exists(self, embed_dim: int) -> None:
        """
        Async version of _create_collection_if_not_exists.

        Args:
            embed_dim: Embedding dimensionality used to initialize default vector
                parameters when not explicitly configured.

        """
        if not self._collection_exists:
            _collection_kwargs = self._prepare_collection_kwargs(embed_dim)
            await self._async_client.collections.create(
                self.collection_name, **_collection_kwargs
            )
            self._collection_exists = True

    def _prepare_collection_kwargs(self, embed_dim: int) -> Dict[str, Any]:
        """
        Build collection creation kwargs, including vectors configuration.

        Args:
            embed_dim: Embedding dimensionality used when deriving default dense
                vector parameters.

        Returns:
            Collection keyword arguments ready for create() calls.

        """
        if self.collection_kwargs is None:
            self.collection_kwargs = {}

        if self.dense_vector_params is None:
            self.dense_vector_params = VectorParams(
                size=embed_dim, distance=Distance.Cosine
            )

        self.collection_kwargs["vectors_config"] = {
            self.dense_vector_name: self.dense_vector_params
        }

        return self.collection_kwargs
