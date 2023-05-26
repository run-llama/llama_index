"""Base query engine."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.response.schema import RESPONSE_TYPE

logger = logging.getLogger(__name__)


class BaseQueryEngine(ABC):
    def __init__(self, callback_manager: Optional[CallbackManager]) -> None:
        self.callback_manager = callback_manager or CallbackManager([])

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            response = self._query(str_or_query_bundle)
            return response

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

    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass
