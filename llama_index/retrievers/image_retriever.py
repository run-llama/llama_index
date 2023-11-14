"""Base vector store index query."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.vector_stores.types import (
    MetadataFilters,
)


class BaseImageRetriever(BaseRetriever):
    """Base Image retriever Abastraction.


    Args:
        BaseImageRetriever
        similarity_top_k (int): number of top k results to return.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
    """

    def __init__(
        self,
        image_similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._image_similarity_top_k = image_similarity_top_k
        self._filters = filters
        self._kwargs: Dict[str, Any] = kwargs

    @property
    def image_similarity_top_k(self) -> int:
        """Return similarity top k."""
        return self._image_similarity_top_k

    @classmethod
    def class_name(cls) -> str:
        return "BaseImageRetriever"

    @abstractmethod
    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        return self._image_retrieve(query_bundle)

    def _image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve image nodes or documents given query.

        Implemented by the user.

        """

    # Async Methods
    @abstractmethod
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await self._aimage_retrieve(query_bundle)

    async def _aimage_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Async retrieve image nodes or documents given query.

        Implemented by the user.

        """
