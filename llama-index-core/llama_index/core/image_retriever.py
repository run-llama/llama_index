from abc import abstractmethod
from typing import List

from llama_index.core.indices.query.schema import QueryBundle, QueryType
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.prompts.mixin import PromptMixin
from llama_index.core.schema import NodeWithScore


class BaseImageRetriever(PromptMixin, DispatcherSpanMixin):
    """Base Image Retriever Abstraction."""

    def text_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        """
        Retrieve image nodes given query or single image input.

        Args:
            str_or_query_bundle (QueryType): a query text
            string or a QueryBundle object.

        """
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(query_str=str_or_query_bundle)
        return self._text_to_image_retrieve(str_or_query_bundle)

    @abstractmethod
    def _text_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """
        Retrieve image nodes or documents given query text.

        Implemented by the user.

        """

    def image_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        """
        Retrieve image nodes given single image input.

        Args:
            str_or_query_bundle (QueryType): a image path
            string or a QueryBundle object.

        """
        if isinstance(str_or_query_bundle, str):
            # leave query_str as empty since we are using image_path for image retrieval
            str_or_query_bundle = QueryBundle(
                query_str="", image_path=str_or_query_bundle
            )
        return self._image_to_image_retrieve(str_or_query_bundle)

    @abstractmethod
    def _image_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """
        Retrieve image nodes or documents given image.

        Implemented by the user.

        """

    # Async Methods
    async def atext_to_image_retrieve(
        self,
        str_or_query_bundle: QueryType,
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(query_str=str_or_query_bundle)
        return await self._atext_to_image_retrieve(str_or_query_bundle)

    @abstractmethod
    async def _atext_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """
        Async retrieve image nodes or documents given query text.

        Implemented by the user.

        """

    async def aimage_to_image_retrieve(
        self,
        str_or_query_bundle: QueryType,
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            # leave query_str as empty since we are using image_path for image retrieval
            str_or_query_bundle = QueryBundle(
                query_str="", image_path=str_or_query_bundle
            )
        return await self._aimage_to_image_retrieve(str_or_query_bundle)

    @abstractmethod
    async def _aimage_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """
        Async retrieve image nodes or documents given image.

        Implemented by the user.

        """
