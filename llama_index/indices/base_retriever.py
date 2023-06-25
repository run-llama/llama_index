from abc import ABC, abstractmethod
from typing import List

from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.query.schema import QueryBundle, QueryType


class BaseRetriever(ABC):
    """Base retriever."""

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Args:
            str_or_query_bundle (QueryType): Either a query string or
                a QueryBundle object.

        """
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._retrieve(str_or_query_bundle)

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        nodes = await self._aretrieve(str_or_query_bundle)
        return nodes

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Implemented by the user.

        """
        pass

    # TODO: make this abstract
    # @abstractmethod
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return []
