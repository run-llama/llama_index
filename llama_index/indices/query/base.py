"""Base query engine."""

import logging
from abc import ABC, abstractmethod
import asyncio
from typing import List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.schema import NodeWithScore

logger = logging.getLogger(__name__)


class BaseQueryEngine(ABC):
    def __init__(self, callback_manager: Optional[CallbackManager]) -> None:
        self.callback_manager = callback_manager or CallbackManager([])

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        return asyncio.get_event_loop().run_until_complete(
            self.aquery(str_or_query_bundle)
        )

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            response = await self._aquery(str_or_query_bundle)
            return response

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError(
            "This query engine does not support retrieve, use query directly"
        )

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError(
            "This query engine does not support aretrieve, use aquery directly"
        )

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError(
            "This query engine does not support synthesize, use query directly"
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError(
            "This query engine does not support asynthesize, use aquery directly"
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return asyncio.get_event_loop().run_until_complete(
            self._aquery(query_bundle)
        )

    @abstractmethod
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass
