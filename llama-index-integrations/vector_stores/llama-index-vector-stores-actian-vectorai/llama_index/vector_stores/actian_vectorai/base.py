"""
Actian Vector AI Vector store index.
"""

from hashlib import sha1
from http import client
from typing import Any, Dict, List, Optional, Union

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

    stores_text: bool = True
    flat_metadata: bool = False

    _client: VectorAIClient = PrivateAttr()
    _collection_name: str = PrivateAttr()

    def __init__(
        self,
        client: VectorAIClient,
        collection_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs)

        if not client.is_connected:
            raise ValueError("ActianVectorAIVectorStore requires a connected VectorAIClient.")

        if not client.collections.exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist in Actian Vector AI.")

        self._client = client
        self._collection_name = collection_name

    @classmethod
    def class_name(cls) -> str:
        return "ActianVectorAIVectorStore"

    @property
    def client(self) -> Any:
        """Return Actian Vector AI client."""
        return self._client

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

        raise NotImplementedError( # Waiting on implementation of scroll method in Actian Vector AI client
            "ActianVectorAIVectorStore.get_nodes() is not implemented."
        )

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        if self._client.collections.exists(self._collection_name) is False:
            raise ValueError(f"Collection '{self._collection_name}' does not exist in Actian Vector AI.")

        points, ids = self._build_points_from_nodes(nodes)
        result = self._client.points.upsert(self._collection_name, points)
        assert result.status == UpdateStatus.Completed, f"Failed to add points to collection {self._collection_name}. Response: {result}"
        return ids


    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The id of the document to delete.

        """
        f = FilterBuilder().must(Field("ref_doc_id").eq(ref_doc_id)).build()
        result = self._client.points.delete(
            self._collection_name,
            filter=f,
        )

        assert result.status == UpdateStatus.Completed, f"Failed to delete points with ref_doc_id {ref_doc_id} from collection {self._collection_name}. Response: {result}"
    
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
        result = self._client.points.delete(
            self._collection_name,
            ids=node_ids,
            filter=self._build_db_filter_from_metadata_filters(filters)
        )

        assert result.status == UpdateStatus.Completed, f"Failed to delete points with ids {node_ids} from collection {self._collection_name}. Response: {result}"

    def clear(self) -> None:
        """
        Clear all nodes from index.
        """
        result = self._client.collections.delete(self._collection_name)
        assert result == True, f"Failed to clear collection {self._collection_name}. Response: {result}"
        
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery object containing query parameters

        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError("Only DEFAULT query mode is supported for ActianVectorAIVectorStore.")
        
        results = self._client.points.search(
            self._collection_name,
            query.query_embedding,
            limit=query.similarity_top_k,
            filter=self._build_db_filter_from_vector_store_query(query),
            **kwargs,
        )
        return self._build_vector_store_query_result_from_scored_points(results)

    def _build_points_from_nodes(self, nodes: List[BaseNode]) -> tuple[List[PointStruct], List[str]]:
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
                    node, remove_text=False, flat_metadata=self.flat_metadata
                )

            point = PointStruct(
                id=node.node_id,
                vector=node.get_embedding(),
                payload=metadata,
            )

            points.append(point)
            ids.append(node.node_id)
        return points, ids
    
    def _build_db_filter_from_metadata_filters(self, filters: MetadataFilters) -> Filter:
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
                conditions.append(Condition(filter=self._build_db_filter_from_metadata_filters(filter)))
            else:
                def filter_operation_to_condition_eq(key: str, value: Any) -> Condition:
                    if isinstance(value, float):
                        return Field(key).between(value, value)
                    else:
                        return Field(key).eq(value)
                    
                def filter_operation_to_condition_ne(key: str, value: Any) -> Condition:
                    if isinstance(value, float):
                        return Condition(filter=FilterBuilder().should(Field(key).lt(value)).should(Field(key).gt(value)).build())
                    else:
                        return Field(key).except_of([value])
                    
                def filter_operation_to_condition_in(key: str, value: Any) -> Condition:
                    if isinstance(value, list):
                        values = value
                    else:
                        values = value.split(",")
                    return Field(key).any_of(values)
                
                def filter_operation_to_condition_nin(key: str, value: Any) -> Condition:
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
                    FilterOperator.TEXT_MATCH: lambda key, value: Field(key).text(value),
                    # FilterOperator.TEXT_MATCH_INSENSITIVE: raise NotImplementedError
                    # FilterOperator.CONTAINS: raise NotImplementedError
                    FilterOperator.IS_EMPTY: lambda key, value: is_empty(key),
                }

                if filter.operator not in fops_dict:
                    raise NotImplementedError(f"Unsupported filter operator: {filter.operator}")
                conditions.append(fops_dict[filter.operator](filter.key, filter.value))

        if filters.condition == FilterCondition.AND:
            return Filter(must = conditions)
        elif filters.condition == FilterCondition.OR:
            return Filter(should = conditions)
        elif filters.condition == FilterCondition.NOT:
            return Filter(must_not = conditions)
        
    def _build_db_filter_from_vector_store_query(self, query: VectorStoreQuery) -> Filter:
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
            conditions.append(Condition(self._build_db_filter_from_metadata_filters(query.filters)))

        return Filter(must=conditions) if conditions else None
    
    def _build_vector_store_query_result_from_scored_points(self, scored_points: List[ScoredPoint]) -> VectorStoreQueryResult:
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