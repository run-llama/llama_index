"""
VDE Vector store index.
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
)

from cortex import CortexClient, DistanceMetric

class VDEVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    flat_metadata: bool = False

    address: str
    api_key: Optional[str]
    pool_size: int
    enable_smart_batching: bool
    batch_size: int
    batch_timeout_ms: int
    timeout: Optional[float]
    collection_name: str
    collection_dimension: int
    distance_metric: Union[DistanceMetric, str]
    hnsw_m: int
    hnsw_ef_construct: int
    hnsw_ef_search: int
    config_json: Optional[str]

    _client: CortexClient = PrivateAttr()

    def __init__(
        self,
        address: str,
        api_key: Optional[str] = None,
        pool_size: int = 3,
        enable_smart_batching: bool = False,  # Disabled by default for sync client
        batch_size: int = 100,
        batch_timeout_ms: int = 100,
        timeout: Optional[float] = None,
        collection_name: str = "llama_index_collection",
        collection_dimension: int = 128,
        distance_metric: Union[DistanceMetric, str] = DistanceMetric.COSINE,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200,
        hnsw_ef_search: int = 50,
        config_json: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            address=address,
            api_key=api_key,
            pool_size=pool_size,
            enable_smart_batching=enable_smart_batching,
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            timeout=timeout,
            collection_name=collection_name,
            collection_dimension=collection_dimension,
            distance_metric=distance_metric,
            hnsw_m=hnsw_m,
            hnsw_ef_construct=hnsw_ef_construct,
            hnsw_ef_search=hnsw_ef_search,
            config_json=config_json,
        )
        self._client = CortexClient(
            address=address,
            api_key=api_key,
            pool_size=pool_size,
            enable_smart_batching=enable_smart_batching,
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            timeout=timeout
        )
        self._client.connect()
        self._client.get_or_create_collection(
            name=collection_name,
            dimension=collection_dimension,
            distance_metric=distance_metric,
            hnsw_m=hnsw_m,
            hnsw_ef_construct=hnsw_ef_construct,
            hnsw_ef_search=hnsw_ef_search,
            config_json=config_json,
        )
        self._client.open_collection(collection_name)

    def __del__(self) -> None:
        if hasattr(self, "_client") and self._client is not None:
            self._client.close_collection(self.collection_name)
            self._client.close()

    @classmethod
    def class_name(cls) -> str:
        return "VDEVectorStore"

    @property
    def client(self) -> Any:
        """Return vde client."""
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
        raise NotImplementedError(
            "VDEVectorStore.add() is not implemented."
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The id of the document to delete.

        """
        raise NotImplementedError(
            "VDEVectorStore.delete() is not implemented."
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery object containing query parameters

        """
        raise NotImplementedError(
            "VDEVectorStore.query() is not implemented."
        )
