from abc import abstractmethod
from typing import List

from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.prompts.mixin import PromptMixin
from llama_index.schema import NodeWithScore


class BaseImageRetriever(PromptMixin):
    """Base Image Retriever Abstraction."""

    def image_retrieve(
        self, str_or_query_bundle: QueryType, is_image_input: bool
    ) -> List[NodeWithScore]:
        """Retrieve image nodes given query or single image input.

        Args:
            str_or_query_bundle (QueryType): Either a query/image_path
            string or a QueryBundle object.
            is_image_input (bool): Whether the input is an image or not.

        """
        if isinstance(str_or_query_bundle, str):
            if is_image_input:
                str_or_query_bundle = QueryBundle(image_path=str_or_query_bundle)
            else:
                str_or_query_bundle = QueryBundle(query_str=str_or_query_bundle)
        return self._image_retrieve(str_or_query_bundle)

    @abstractmethod
    def _image_retrieve(
        self,
        query_bundle: QueryBundle,
        is_image_input: bool,
    ) -> List[NodeWithScore]:
        """Retrieve image nodes or documents given query or image.

        Implemented by the user.

        """

    # Async Methods
    async def aimage_retrieve(
        self, str_or_query_bundle: QueryType, is_image_input: bool
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            if is_image_input:
                str_or_query_bundle = QueryBundle(image_path=str_or_query_bundle)
            else:
                str_or_query_bundle = QueryBundle(query_str=str_or_query_bundle)
        return await self._aimage_retrieve(str_or_query_bundle)

    # @abstractmethod
    async def _aimage_retrieve(
        self,
        query_bundle: QueryBundle,
        is_image_input: bool,
    ) -> List[NodeWithScore]:
        """Async retrieve image nodes or documents given query or image.

        Implemented by the user.

        """
        # return self._image_retrieve(query_bundle, is_image_input)
