"""
S3 vector store index.

An index that is built on top of an existing S3Vectors collection.

"""

import asyncio
import boto3
import logging
import time
from typing import Any, List, Optional, Tuple

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)


class S3VectorStore(BasePydanticVectorStore):
    """
    S3 Vector Store.

    Uses the S3Vectors service to store and query vectors directly in S3.

    It is recommended to create a vector bucket in S3 first.

    Args:
        index_name (str): The name of the index.
        bucket_name_or_arn (str): The name or ARN of the vector bucket.
        data_type (str): The data type of the vectors. Only supports "float32" for now.
        insert_batch_size (int): The batch size for inserting vectors.
        sync_session (Optional[boto3.Session]): The session to use for the synchronous client.

    Examples:
        `pip install llama-index-vector-stores-s3`

        ```python
        from llama_index.vector_stores.s3 import S3VectorStore

        vector_store = S3VectorStore.create_index_from_bucket(
            bucket_name_or_arn="my-vector-bucket",
            index_name="my-index",
            dimension=1536,
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = False

    index_name_or_arn: str = Field(description="The name or ARN of the index.")
    bucket_name_or_arn: str = Field(description="The name or ARN of the bucket.")
    data_type: str = Field(description="The data type of the vectors.")
    insert_batch_size: int = Field(description="The batch size for inserting vectors.")
    text_field: Optional[str] = Field(
        default=None, description="The field to use as the text field in the metadata."
    )
    distance_metric: str = Field(
        default="cosine", description="The distance metric used by the index."
    )

    _session: boto3.Session = PrivateAttr()

    def __init__(
        self,
        index_name_or_arn: str,
        bucket_name_or_arn: str,
        data_type: str = "float32",
        insert_batch_size: int = 500,
        text_field: Optional[str] = None,
        distance_metric: str = "cosine",
        sync_session: Optional[boto3.Session] = None,
        async_session: Optional[Any] = None,
    ) -> None:
        """Init params."""
        if async_session is not None:
            raise NotImplementedError(
                "Async sessions are not supported yet by aioboto3/aiobotocore"
            )

        if insert_batch_size > 500:
            raise ValueError("Insert batch size must be less than or equal to 500")

        super().__init__(
            index_name_or_arn=index_name_or_arn,
            bucket_name_or_arn=bucket_name_or_arn,
            data_type=data_type,
            insert_batch_size=insert_batch_size,
            text_field=text_field,
            distance_metric=distance_metric,
        )
        self._session = sync_session or boto3.Session()

    @classmethod
    def create_index_from_bucket(
        cls,
        bucket_name_or_arn: str,
        index_name: str,
        dimension: int,
        distance_metric: str = "cosine",
        data_type: str = "float32",
        insert_batch_size: int = 500,
        non_filterable_metadata_keys: Optional[List[str]] = None,
        sync_session: Optional[boto3.Session] = None,
        async_session: Optional[Any] = None,
    ) -> "S3VectorStore":
        """
        Create an index in S3Vectors.
        """
        # node content and node type should never be filterable by default
        non_filterable_metadata_keys = non_filterable_metadata_keys or []
        if "_node_content" not in non_filterable_metadata_keys:
            non_filterable_metadata_keys.append("_node_content")
        if "_node_type" not in non_filterable_metadata_keys:
            non_filterable_metadata_keys.append("_node_type")

        bucket_name, bucket_arn = cls.get_name_or_arn(bucket_name_or_arn)

        sync_session = sync_session or boto3.Session()
        kwargs = {
            "indexName": index_name,
            "dimension": dimension,
            "dataType": data_type,
            "distanceMetric": distance_metric,
            "metadataConfiguration": {
                "nonFilterableMetadataKeys": non_filterable_metadata_keys,
            },
        }
        if bucket_arn is not None:
            kwargs["vectorBucketArn"] = bucket_arn
        else:
            kwargs["vectorBucketName"] = bucket_name

        sync_session.client("s3vectors").create_index(**kwargs)

        return cls(
            sync_session=sync_session,
            async_session=async_session,
            data_type=data_type,
            index_name_or_arn=index_name,
            bucket_name_or_arn=bucket_name_or_arn,
            insert_batch_size=insert_batch_size,
            distance_metric=distance_metric,
        )

    @classmethod
    def class_name(cls) -> str:
        return "S3VectorStore"

    @staticmethod
    def get_name_or_arn(name_or_arn: str) -> Tuple[str, str]:
        """
        Get the name or ARN.
        """
        if "arn:" in name_or_arn:
            return None, name_or_arn
        return name_or_arn, None

    def _parse_response(self, response: dict) -> List[BaseNode]:
        """
        Parse the response from S3Vectors.
        """
        if self.text_field is None:
            return [
                metadata_dict_to_node(v["metadata"])
                for v in response.get("vectors", [])
            ]
        else:
            nodes = []
            for v in response.get("vectors", []):
                if self.text_field not in v["metadata"]:
                    raise ValueError(
                        f"Text field {self.text_field} not found in returned metadata"
                    )

                text = v["metadata"].pop(self.text_field)
                nodes.append(TextNode(text=text, metadata=v["metadata"]))
            return nodes

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """
        Get nodes from the index.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to retrieve.
            filters (Optional[MetadataFilters]): Metadata filters to apply.

        Returns:
            List[BaseNode]: List of nodes retrieved from the index.

        """
        if node_ids is None:
            raise ValueError("node_ids is required")

        if filters is not None:
            raise NotImplementedError("Filters are not supported yet")

        index_name, index_arn = self.get_name_or_arn(self.index_name_or_arn)

        kwargs = {
            "keys": node_ids,
            "vectorBucketName": self.bucket_name_or_arn,
            "returnMetadata": True,
        }
        if index_arn is not None:
            kwargs["indexArn"] = index_arn
        else:
            kwargs["indexName"] = index_name

        response = self._session.client("s3vectors").get_vectors(**kwargs)
        return self._parse_response(response)

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """
        Asynchronous method to get nodes from the index.

        Args:
            node_ids (Optional[List[str]]): List of node IDs to retrieve.
            filters (Optional[MetadataFilters]): Metadata filters to apply.

        Returns:
            List[BaseNode]: List of nodes retrieved from the index.

        """
        return await asyncio.to_thread(
            self.get_nodes, node_ids=node_ids, filters=filters
        )

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        index_name, index_arn = self.get_name_or_arn(self.index_name_or_arn)

        # limit to 5 requests per second
        # Poor-mans token bucket
        start_time = time.time()
        available_requests = 5
        added_ids = []
        for node_batch in iter_batch(nodes, self.insert_batch_size):
            vectors = []
            for node in node_batch:
                node_metadata = node_to_metadata_dict(node)

                # delete fields that aren't used to save space
                node_metadata.pop("document_id", None)
                node_metadata.pop("doc_id", None)
                node_metadata.pop("embedding", None)

                vectors.append(
                    {
                        "key": str(node.id_),
                        "data": {"float32": node.embedding},
                        "metadata": {**node_metadata},
                    }
                )

            kwargs = {
                "vectors": vectors,
                "vectorBucketName": self.bucket_name_or_arn,
            }
            if index_arn is not None:
                kwargs["indexArn"] = index_arn
            else:
                kwargs["indexName"] = index_name

            self._session.client("s3vectors").put_vectors(**kwargs)

            # Update the token bucket
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                available_requests = 5
            else:
                available_requests -= 1

            if available_requests == 0:
                time.sleep(1)
                available_requests = 5

            added_ids.extend([v["key"] for v in vectors])

        return added_ids

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
        return await asyncio.to_thread(self.add, nodes, **kwargs)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
            delete_kwargs (Any): Additional arguments to pass to the list_vectors method.

        """
        index_name, index_arn = self.get_name_or_arn(self.index_name_or_arn)
        next_token = None
        while True:
            kwargs = {
                "vectorBucketName": self.bucket_name_or_arn,
                "returnMetadata": True,
                **delete_kwargs,
            }
            if index_arn is not None:
                kwargs["indexArn"] = index_arn
            else:
                kwargs["indexName"] = index_name

            response = self._session.client("s3vectors").list_vectors(**kwargs)

            nodes_to_delete = [
                v["key"]
                for v in response.get("vectors", [])
                if v["metadata"]["ref_doc_id"] == ref_doc_id
            ]

            kwargs = {
                "vectorBucketName": self.bucket_name_or_arn,
                "keys": nodes_to_delete,
            }
            if index_arn is not None:
                kwargs["indexArn"] = index_arn
            else:
                kwargs["indexName"] = index_name

            self._session.client("s3vectors").delete_vectors(**kwargs)

            next_token = response.get("nextToken")
            if next_token is None:
                break

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Asynchronous method to delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        return await asyncio.to_thread(self.delete, ref_doc_id, **delete_kwargs)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes using with node_ids.

        Args:
            node_ids (Optional[List[str]): List of node IDs to delete.
            filters (Optional[MetadataFilters]): Metadata filters to apply.

        """
        if filters is not None:
            raise NotImplementedError("Deleting by filters is not supported yet")

        if node_ids is None:
            raise ValueError("node_ids is required")

        index_name, index_arn = self.get_name_or_arn(self.index_name_or_arn)
        kwargs = {
            "vectorBucketName": self.bucket_name_or_arn,
            "keys": node_ids,
        }
        if index_arn is not None:
            kwargs["indexArn"] = index_arn
        else:
            kwargs["indexName"] = index_name
        self._session.client("s3vectors").delete_vectors(**kwargs)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Asynchronous method to delete nodes using with node_ids.

        Args:
            node_ids (Optional[List[str]): List of node IDs to delete.
            filters (Optional[MetadataFilters]): Metadata filters to apply.

        """
        return await asyncio.to_thread(
            self.delete_nodes, node_ids=node_ids, filters=filters, **delete_kwargs
        )

    def clear(self) -> None:
        """
        Clear the index.
        """
        index_name, index_arn = self.get_name_or_arn(self.index_name_or_arn)
        kwargs = {
            "vectorBucketName": self.bucket_name_or_arn,
        }
        if index_arn is not None:
            kwargs["indexArn"] = index_arn
        else:
            kwargs["indexName"] = index_name
        self._session.client("s3vectors").delete_index(**kwargs)

    async def aclear(self) -> None:
        """
        Asynchronous method to clear the index.
        """
        return await asyncio.to_thread(self.clear)

    @property
    def client(self) -> Any:
        """Return the Qdrant client."""
        return self._session.client("s3vectors")

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
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(
                "Only DEFAULT query mode is supported for S3VectorStore"
            )

        index_name, index_arn = self.get_name_or_arn(self.index_name_or_arn)
        kwargs = {
            "vectorBucketName": self.bucket_name_or_arn,
            "queryVector": {self.data_type: query.query_embedding},
            "topK": query.similarity_top_k,
            "filter": self._build_filter(query.filters),
            "returnDistance": True,
            "returnMetadata": True,
        }
        if index_arn is not None:
            kwargs["indexArn"] = index_arn
        else:
            kwargs["indexName"] = index_name
        response = self._session.client("s3vectors").query_vectors(**kwargs)

        nodes = self._parse_response(response)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=self._convert_distances_to_similarities(
                response.get("vectors", [])
            ),
            ids=[v["key"] for v in response.get("vectors", [])],
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Asynchronous method to query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): query

        """
        return await asyncio.to_thread(self.query, query, **kwargs)

    def _build_filter(self, filters: Optional[MetadataFilters]) -> Optional[dict]:
        """
        Build a filter for the query.
        """
        if filters is None:
            return None

        def _convert_single_filter(filter_obj) -> dict:
            """Convert a single MetadataFilter to S3 Vectors format."""
            if not isinstance(filter_obj, MetadataFilter):
                raise ValueError(f"Expected MetadataFilter, got {type(filter_obj)}")

            key = filter_obj.key
            value = filter_obj.value
            operator = filter_obj.operator

            # Map LlamaIndex operators to S3 Vectors operators
            operator_map = {
                FilterOperator.EQ: "$eq",
                FilterOperator.NE: "$ne",
                FilterOperator.GT: "$gt",
                FilterOperator.GTE: "$gte",
                FilterOperator.LT: "$lt",
                FilterOperator.LTE: "$lte",
                FilterOperator.IN: "$in",
                FilterOperator.NIN: "$nin",
            }

            if operator == FilterOperator.IS_EMPTY:
                # For IS_EMPTY, we use $exists with false
                return {key: {"$exists": False}}
            elif operator in operator_map:
                return {key: {operator_map[operator]: value}}
            else:
                # Unsupported operators - for now, we'll raise an error
                # Could potentially map TEXT_MATCH, ANY, ALL, CONTAINS if S3 Vectors supports them
                raise ValueError(f"Unsupported filter operator: {operator}")

        def _convert_filters_recursively(filters_obj) -> dict:
            """Recursively convert MetadataFilters to S3 Vectors format."""
            if isinstance(filters_obj, MetadataFilter):
                return _convert_single_filter(filters_obj)
            elif isinstance(filters_obj, MetadataFilters):
                filter_list = []

                for f in filters_obj.filters:
                    converted_filter = _convert_filters_recursively(f)
                    filter_list.append(converted_filter)

                # Handle the condition
                if len(filter_list) == 1:
                    return filter_list[0]
                elif filters_obj.condition == FilterCondition.AND:
                    return {"$and": filter_list}
                elif filters_obj.condition == FilterCondition.OR:
                    return {"$or": filter_list}
                elif filters_obj.condition == FilterCondition.NOT:
                    # S3 Vectors doesn't have explicit $not
                    # We would need to implement a custom filter that negates the logic
                    raise ValueError(
                        "NOT condition is not supported for S3 Vectors filters"
                    )
                else:
                    raise ValueError(
                        f"Unexpected filter condition: {filters_obj.condition}"
                    )
            else:
                raise ValueError(f"Unexpected filter type: {type(filters_obj)}")

        return _convert_filters_recursively(filters)

    def _convert_distances_to_similarities(self, vectors: List[dict]) -> List[float]:
        """
        Convert distances to similarity scores (0-1 scale, where 1 is most similar).

        Args:
            vectors: List of vector results containing distance values

        Returns:
            List of similarity scores normalized to [0, 1] where 1 is most similar

        """
        similarities = []

        for vector in vectors:
            distance = float(vector.get("distance", 0))

            if self.distance_metric.lower() == "cosine":
                # Cosine distance is typically in range [0, 2] where 0 is most similar
                # Convert to similarity: similarity = 1 - (distance / 2)
                # But if distance is already normalized to [0, 1], use: similarity = 1 - distance
                similarity = max(0.0, min(1.0, 1.0 - distance))
            elif self.distance_metric.lower() == "euclidean":
                # Euclidean distance ranges from 0 to infinity
                # Use: similarity = 1 / (1 + distance) which maps [0, ∞) to (0, 1]
                similarity = 1.0 / (1.0 + distance)
            else:
                # For unknown metrics, assume cosine-like behavior
                similarity = max(0.0, min(1.0, 1.0 - distance))

            similarities.append(similarity)

        return similarities
