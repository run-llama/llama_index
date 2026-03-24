"""
Actian Vector AI Vector store index.
"""

from __future__ import annotations

from hashlib import sha1
from http import client
from typing import Any, Dict, List, Optional, Sequence, Union
import asyncio

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
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from actian_vectorai import (
    Condition,
    Field,
    FilterBuilder,
    HnswConfigDiff,
    VectorAIClient,
    AsyncVectorAIClient,
    WalConfigDiff,
    Filter,
    is_empty,
    has_id,
)

from actian_vectorai.models import (
    Distance,
    IndexType,
    OptimizersConfigDiff,
    QuantizationConfig,
    PointStruct,
    ScoredPoint,
    ShardingMethod,
    UpdateResult,
    UpdateStatus,
    VectorParams,
)


class ActianVectorAIVectorStore(BasePydanticVectorStore):

    stores_text: bool = False
    flat_metadata: bool = False

    url: str
    collection_name: str
    client_kwargs: Optional[Dict[str, Any]]
    collection_kwargs: Optional[Dict[str, Any]]

    _client: VectorAIClient = PrivateAttr(None)
    _async_client: AsyncVectorAIClient = PrivateAttr(None)
    _collection_exists: bool = PrivateAttr(False)
    _clear_existing_collection: bool = PrivateAttr(False)
    _lazy_collection_create: bool = PrivateAttr(False)

    def __init__(
        self,
        url: str = "localhost:50051",
        collection_name: str = "llama_index_collection",
        client_kwargs: Optional[dict[str, Any]] = None,
        collection_kwargs: Optional[dict[str, Any]] = None,
        client: Optional[VectorAIClient] = None,
        async_client: Optional[AsyncVectorAIClient] = None,
        clear_existing_collection: bool = False,
        stores_text: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            url=url,
            collection_name=collection_name,
            client_kwargs=client_kwargs,
            collection_kwargs=collection_kwargs,
            stores_text=stores_text,
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

    def __enter__(self) -> ActianVectorAIVectorStore:
        if self._client is None:
            raise ValueError(
                "Synchronous client is not initialized. Please initialize the ActianVectorAIVectorStore with a VectorAIClient to use synchronous methods."
            )
        if not self._client.is_connected:
            self._client.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._client.is_connected:
            self._client.shutdown()

    async def __aenter__(self) -> ActianVectorAIVectorStore:
        if self._async_client is None:
            raise ValueError(
                "Async client is not initialized. Please initialize the ActianVectorAIVectorStore with an AsyncVectorAIClient to use asynchronous methods."
            )
        if not self._async_client.is_connected:
            await self._async_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
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
    ) -> List[BaseNode]:
        """
        Get nodes by ids or metadata filters.

        Args:
            node_ids (List[str]): List of node ids to get.
            filters (MetadataFilters): Metadata filters to apply to query.
        Returns:
            List[BaseNode]: List of nodes matching query.
        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return []

        raise NotImplementedError(  # Waiting on implementation of scroll method in Actian Vector AI client
            "ActianVectorAIVectorStore.get_nodes() is not implemented."
        )

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Asynchronously get nodes from vector store."""
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return []

        raise NotImplementedError(  # Waiting on implementation of scroll method in Actian Vector AI client
            "ActianVectorAIVectorStore.aget_nodes() is not implemented."
        )

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        self._lazy_client_operation_check()

        if not len(nodes) > 0:
            return []

        self._create_collection_if_not_exists(len(nodes[0].get_embedding()))

        points, ids = self._build_points_from_nodes(nodes)
        result = self._client.points.upsert(self.collection_name, points)
        assert (
            result.status == UpdateStatus.Completed
        ), f"Failed to add points to collection {self.collection_name}. Response: {result}"
        return ids

    async def aadd(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Asynchronously add nodes to index."""
        await self._lazy_async_client_operation_check()

        if not len(nodes) > 0:
            return []

        await self._acreate_collection_if_not_exists(len(nodes[0].get_embedding()))

        points, ids = self._build_points_from_nodes(nodes)
        result = await self._async_client.points.upsert(self.collection_name, points)
        assert (
            result.status == UpdateStatus.Completed
        ), f"Failed to add points to collection {self.collection_name}. Response: {result}"
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The id of the document to delete.

        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return

        result = self._client.points.delete(
            self.collection_name,
            filter=FilterBuilder().must(Field("ref_doc_id").eq(ref_doc_id)).build(),
        )

        assert (
            result.status == UpdateStatus.Completed
        ), f"Failed to delete points with ref_doc_id {ref_doc_id} from collection {self.collection_name}. Response: {result}"

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Asynchronously delete nodes using with ref_doc_id."""
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return

        result = await self._async_client.points.delete(
            self.collection_name,
            filter=FilterBuilder().must(Field("ref_doc_id").eq(ref_doc_id)).build(),
        )

        assert (
            result.status == UpdateStatus.Completed
        ), f"Failed to delete points with ref_doc_id {ref_doc_id} from collection {self.collection_name}. Response: {result}"

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes using list of node ids.

        Args:
            node_ids (List[str]): The list of node ids to delete.

        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return

        result = self._client.points.delete(
            self.collection_name,
            ids=node_ids,
            filter=self._build_db_filter_from_metadata_filters(filters),
        )

        assert (
            result.status == UpdateStatus.Completed
        ), f"Failed to delete points from collection {self.collection_name}. Response: {result}"

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronously delete nodes using list of node ids."""
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return

        result = await self._async_client.points.delete(
            self.collection_name,
            ids=node_ids,
            filter=self._build_db_filter_from_metadata_filters(filters),
        )

        assert (
            result.status == UpdateStatus.Completed
        ), f"Failed to delete points with ids {node_ids} from collection {self.collection_name}. Response: {result}"

    def clear(self) -> None:
        """
        Clear all nodes from index.
        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return

        result = self._client.collections.delete(self.collection_name)
        assert (
            result == True
        ), f"Failed to clear collection {self.collection_name}. Response: {result}"

        self._collection_exists = False

    async def aclear(self) -> None:
        """
        Clear all nodes from index.
        """
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return

        result = await self._async_client.collections.delete(self.collection_name)
        assert (
            result == True
        ), f"Failed to clear collection {self.collection_name}. Response: {result}"

        self._collection_exists = False

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery object containing query parameters

        """
        self._lazy_client_operation_check()

        if not self._collection_exists:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(
                "Only DEFAULT query mode is supported for ActianVectorAIVectorStore."
            )

        results = self._client.points.search(
            self.collection_name,
            query.query_embedding,
            limit=query.similarity_top_k,
            filter=self._build_db_filter_from_vector_store_query(query),
            **kwargs,
        )
        return self._build_vector_store_query_result_from_scored_points(results)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Asynchronously query index for top k most similar nodes."""
        await self._lazy_async_client_operation_check()

        if not self._collection_exists:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(
                "Only DEFAULT query mode is supported for ActianVectorAIVectorStore."
            )

        results = await self._async_client.points.search(
            self.collection_name,
            query.query_embedding,
            limit=query.similarity_top_k,
            filter=self._build_db_filter_from_vector_store_query(query),
            **kwargs,
        )
        return self._build_vector_store_query_result_from_scored_points(results)

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
        Build list of points to add to Actian Vector AI collection from list of nodes.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        Returns:
            tuple[List[PointStruct], List[str]]: list of points to add to Actian Vector AI collection and their corresponding IDs
        """
        points = []
        ids = []

        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=not self.stores_text, flat_metadata=self.flat_metadata
            )

            point = PointStruct(
                id=node.node_id,
                vector=node.get_embedding(),
                payload=metadata,
            )

            points.append(point)
            ids.append(node.node_id)
        return points, ids

    def _build_db_filter_from_metadata_filters(
        self, filters: MetadataFilters
    ) -> Filter:
        """
        Build Actian Vector AI filter from LlamaIndex MetadataFilters.

        Args:
            filters: MetadataFilters object containing list of filter groups
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
                    # FilterOperator.ANY: raise NotImplementedError
                    # FilterOperator.ALL: raise NotImplementedError
                    FilterOperator.TEXT_MATCH: lambda key, value: Field(key).text(
                        value
                    ),
                    # FilterOperator.TEXT_MATCH_INSENSITIVE: raise NotImplementedError
                    # FilterOperator.CONTAINS: raise NotImplementedError
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

    def _build_db_filter_from_vector_store_query(
        self, query: VectorStoreQuery
    ) -> Filter:
        """
        Build Actian Vector AI filter from LlamaIndex VectorStoreQuery.

        Args:
            query: VectorStoreQuery object containing query parameters
        """
        conditions = []

        if query.node_ids is not None:
            conditions.append(has_id(query.node_ids))

        if query.doc_ids is not None:
            conditions.append(Field("ref_doc_id").any_of(query.doc_ids))

        if query.filters is not None:
            conditions.append(
                Condition(self._build_db_filter_from_metadata_filters(query.filters))
            )

        return Filter(must=conditions) if conditions else None

    def _build_vector_store_query_result_from_scored_points(
        self, scored_points: List[ScoredPoint]
    ) -> VectorStoreQueryResult:
        """
        Build LlamaIndex VectorStoreQueryResult from list of Actian Vector AI ScoredPoint.

        Args:
            scored_points: List of ScoredPoints returned from Actian Vector AI search query
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

    def _lazy_client_operation_check(self) -> None:
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

        if not self._lazy_collection_create:
            if self._client.collections.exists(self.collection_name):
                self._collection_exists = True
            elif self.collection_kwargs is not None:
                self._client.collections.create(
                    self.collection_name, **self.collection_kwargs or {}
                )
                self._collection_exists = True
            else:
                self._collection_exists = False
            self._lazy_collection_create = True

    async def _lazy_async_client_operation_check(self) -> None:
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

        if not self._lazy_collection_create:
            if await self._async_client.collections.exists(self.collection_name):
                self._collection_exists = True
            elif self.collection_kwargs is not None:
                await self._async_client.collections.create(
                    self.collection_name, **self.collection_kwargs or {}
                )
                self._collection_exists = True
            else:
                self._collection_exists = False
            self._lazy_collection_create = True

    def _create_collection_if_not_exists(self, embed_dim: int) -> None:
        if not self._collection_exists:
            if self.collection_kwargs is not None:
                self._client.collections.create(
                    self.collection_name, **self.collection_kwargs or {}
                )
            else:
                # Default to creating collection with cosine distance and HNSW index if no collection kwargs provided
                self._client.collections.create(
                    self.collection_name,
                    vectors_config=VectorParams(
                        size=embed_dim, distance=Distance.Cosine
                    ),
                )
        self._collection_exists = True

    async def _acreate_collection_if_not_exists(self, embed_dim: int) -> None:
        if not self._collection_exists:
            if self.collection_kwargs is not None:
                await self._async_client.collections.create(
                    self.collection_name, **self.collection_kwargs or {}
                )
            else:
                # Default to creating collection with cosine distance and HNSW index if no collection kwargs provided
                await self._async_client.collections.create(
                    self.collection_name,
                    vectors_config=VectorParams(
                        size=embed_dim, distance=Distance.Cosine
                    ),
                )
        self._collection_exists = True
