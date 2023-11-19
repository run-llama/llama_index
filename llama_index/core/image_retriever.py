from abc import abstractmethod
from typing import List

from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.prompts.mixin import PromptMixin
from llama_index.schema import NodeWithScore


class BaseImageRetriever(PromptMixin):
    """Base Image Retriever Abstraction."""

    def image_retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve image nodes given query or single image input.

        Args:
            str_or_query_bundle (QueryType): Either a query/image_path
            string or a QueryBundle object.
        """
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(query_str=str_or_query_bundle)
        return self._image_retrieve(str_or_query_bundle)

    @abstractmethod
    def _image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve image nodes or documents given query or image.

        Implemented by the user.

        """

    # Async Methods
    async def aimage_retrieve(
        self,
        str_or_query_bundle: QueryType,
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(query_str=str_or_query_bundle)
        return await self._aimage_retrieve(str_or_query_bundle)

    @abstractmethod
    async def _aimage_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Async retrieve image nodes or documents given query or image.

        Implemented by the user.

        """
