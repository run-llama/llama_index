"""base multi modal retriever."""
from abc import abstractmethod
from typing import List

from llama_index.core.base_retriever import BaseRetriever
from llama_index.core.image_retriever import BaseImageRetriever
from llama_index.indices.query.schema import QueryType
from llama_index.schema import NodeWithScore


class MultiModalRetriever(BaseRetriever, BaseImageRetriever):
    """Multi Modal base retriever."""

    @abstractmethod
    def text_retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve text nodes given query.

        Implemented by the user.

        """

    @abstractmethod
    def image_retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve image nodes given query.

        Implemented by the user.

        """

    @abstractmethod
    async def atext_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        """Async Retrieve text nodes given query.

        Implemented by the user.

        """

    @abstractmethod
    async def aimage_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        """Async Retrieve image nodes given query.

        Implemented by the user.

        """
