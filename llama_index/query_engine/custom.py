"""Custom query engine."""

from llama_index.indices.query.base import BaseQueryEngine
from typing import Optional, Any
from llama_index.callbacks.base import CallbackManager
from llama_index.response.schema import Response
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.response.schema import RESPONSE_TYPE
from abc import abstractmethod


class CustomQueryEngine(BaseQueryEngine):
    """Custom query engine.

    Subclasses must implement the `init_params`, `custom_query` functions.

    Optionally, you can implement the async version `acustom_query` as well.

    Optionally, you can also implement the `callback_manager` property to
        return a CallbackManager instance.
    
    """

    def __init__(
        self, 
        callback_manager: Optional[CallbackManager] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Initialize the custom query engine."""
        super().__init__(callback_manager=callback_manager)
        self.init_params(*args, **kwargs)

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            response = self.custom_query(str_or_query_bundle)
            return response

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            response = await self.acustom_query(str_or_query_bundle)
            return response

    @abstractmethod
    def init_params(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the custom query engine.

        Treat it as you would defining the __init__ function of a class.
        
        """

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
