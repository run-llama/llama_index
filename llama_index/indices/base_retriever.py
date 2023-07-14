from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.schema import NodeWithScore
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.indices.service_context import ServiceContext


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

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Implemented by the user.

        """
        pass

    def get_service_context(self) -> Optional[ServiceContext]:
        """Attempts to resolve a service context.
        Short-circuits at self.service_context, self._service_context,
        or self._index.service_context
        """
        if hasattr(self, "service_context"):
            return self.service_context
        if hasattr(self, "_service_context"):
            return self._service_context
        elif hasattr(self, "_index") and hasattr(self._index, "service_context"):
            return self._index.service_context
        return None
