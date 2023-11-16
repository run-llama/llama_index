"""Custom query engine."""

from abc import abstractmethod
from typing import Union

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.callbacks.base import CallbackManager
from llama_index.core import BaseQueryEngine
from llama_index.prompts.mixin import PromptMixinType
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.schema import QueryBundle, QueryType

STR_OR_RESPONSE_TYPE = Union[Response, str]


class CustomQueryEngine(BaseModel, BaseQueryEngine):
    """Custom query engine.

    Subclasses can define additional attributes as Pydantic fields.
    Subclasses must implement the `custom_query` method, which takes a query string
    and returns either a Response object or a string as output.

    They can optionally implement the `acustom_query` method for async support.

    """

    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    class Config:
        arbitrary_types_allowed = True

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            # if query bundle, just run the query
            if isinstance(str_or_query_bundle, QueryBundle):
                query_str = str_or_query_bundle.query_str
            else:
                query_str = str_or_query_bundle
            raw_response = self.custom_query(query_str)
            return (
                Response(raw_response)
                if isinstance(raw_response, str)
                else raw_response
            )

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, QueryBundle):
                query_str = str_or_query_bundle.query_str
            else:
                query_str = str_or_query_bundle
            raw_response = await self.acustom_query(query_str)
            return (
                Response(raw_response)
                if isinstance(raw_response, str)
                else raw_response
            )

    @abstractmethod
    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        """Run a custom query."""

    async def acustom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        """Run a custom query asynchronously."""
        # by default, just run the synchronous version
        return self.custom_query(query_str)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        raise NotImplementedError("This query engine does not support _query.")

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        raise NotImplementedError("This query engine does not support _aquery.")
