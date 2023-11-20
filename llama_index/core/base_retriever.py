"""Base retriever."""
from abc import abstractmethod
from typing import List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.mixin import PromptDictType, PromptMixin, PromptMixinType
from llama_index.schema import NodeWithScore


class BaseRetriever(PromptMixin):
    """Base retriever."""

    def __init__(self, callback_manager: Optional[CallbackManager]) -> None:
        self.callback_manager = callback_manager or CallbackManager([])

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Args:
            str_or_query_bundle (QueryType): Either a query string or
                a QueryBundle object.

        """
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(query_bundle)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Implemented by the user.

        """

    # TODO: make this abstract
    # @abstractmethod
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return self._retrieve(query_bundle)

    def get_service_context(self) -> Optional[ServiceContext]:
        """Attempts to resolve a service context.
        Short-circuits at self.service_context, self._service_context,
        or self._index.service_context.
        """
        if hasattr(self, "service_context"):
            return self.service_context
        if hasattr(self, "_service_context"):
            return self._service_context
        elif hasattr(self, "_index") and hasattr(self._index, "service_context"):
            return self._index.service_context
        return None
