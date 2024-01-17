from abc import abstractmethod
from typing import Any, List, Tuple

from llama_index.bridge.pydantic import BaseModel
from llama_index.core.base_retriever import BaseRetriever
from llama_index.schema import NodeWithScore, QueryBundle


class BaseAutoRetriever(BaseRetriever):
    """Base auto retriever."""

    @abstractmethod
    def generate_retrieval_spec(
        self, query_bundle: QueryBundle, **kwargs: Any
    ) -> BaseModel:
        """Generate retrieval spec synchronously."""
        ...

    @abstractmethod
    async def agenerate_retrieval_spec(
        self, query_bundle: QueryBundle, **kwargs: Any
    ) -> BaseModel:
        """Generate retrieval spec asynchronously."""
        ...

    @abstractmethod
    def _build_retriever_from_spec(
        self, retrieval_spec: BaseModel
    ) -> Tuple[BaseRetriever, QueryBundle]:
        """Build retriever from spec and provide query bundle."""
        ...

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using generated spec."""
        retrieval_spec = self.generate_retrieval_spec(query_bundle)
        retriever, new_query_bundle = self._build_retriever_from_spec(retrieval_spec)
        return retriever.retrieve(new_query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using generated spec asynchronously."""
        retrieval_spec = await self.agenerate_retrieval_spec(query_bundle)
        retriever, new_query_bundle = self._build_retriever_from_spec(retrieval_spec)
        return await retriever.aretrieve(new_query_bundle)
