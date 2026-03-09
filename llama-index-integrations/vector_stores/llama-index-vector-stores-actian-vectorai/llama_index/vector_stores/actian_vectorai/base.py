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
)

from actian_vectorai import (
    HnswConfigDiff, 
    VectorAIClient, 
    WalConfigDiff
)

from actian_vectorai.models import (
    Distance,
    IndexType,
    OptimizersConfigDiff,
    QuantizationConfig,
    ShardingMethod,
    VectorParams,
)

class ActianVectorAIVectorStore(BasePydanticVectorStore):

    stores_text: bool = True

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
        raise NotImplementedError(
            "ActianVectorAIVectorStore.add() is not implemented."
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The id of the document to delete.

        """
        raise NotImplementedError(
            "ActianVectorAIVectorStore.delete() is not implemented."
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery object containing query parameters

        """
        raise NotImplementedError(
            "ActianVectorAIVectorStore.query() is not implemented."
        )
