"""Custom query engine."""

from llama_index.indices.query.base import BaseQueryEngine
from typing import Optional, Any
from pydantic import BaseModel
from llama_index.callbacks.base import CallbackManager
from llama_index.response.schema import Response
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.bridge.pydantic import Field
from abc import abstractmethod


class CustomQueryEngine(BaseModel, BaseQueryEngine):
    """Custom query engine.
    
    """

    callback_manager: Optional[CallbackManager] = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )

    class Config:
        arbitrary_types_allowed = True
  
    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            # if query bundle, just run the query
            if isinstance(str_or_query_bundle, QueryBundle):
                query_str = str_or_query_bundle.query_str
            else:
                query_str = str_or_query_bundle
            response = self.custom_query(query_str)
            return response

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, QueryBundle):
                query_str = str_or_query_bundle.query_str
            else:
                query_str = str_or_query_bundle
            response = await self.acustom_query(query_str)
            return response

    @abstractmethod
    def custom_query(self, query_str: str) -> Response:
        """Run a custom query."""

    async def acustom_query(self, query_str: str) -> Response:
        """Run a custom query asynchronously."""
        # by default, just run the synchronous version
        return self.custom_query(query_str)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass
