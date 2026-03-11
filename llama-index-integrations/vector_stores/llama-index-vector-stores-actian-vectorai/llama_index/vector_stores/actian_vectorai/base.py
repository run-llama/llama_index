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
    Field,
    FilterBuilder,
    HnswConfigDiff, 
    VectorAIClient, 
    WalConfigDiff,
)

from actian_vectorai.models import (
    Distance,
    IndexType,
    OptimizersConfigDiff,
    QuantizationConfig,
    PointStruct,
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

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery object containing query parameters

        """
        raise NotImplementedError(
            "ActianVectorAIVectorStore.query() is not implemented."
        )

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