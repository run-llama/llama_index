"""Base query engine."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence
from llama_index.data_structs.node import NodeWithScore

from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.response.schema import (
    RESPONSE_TYPE,
)

logger = logging.getLogger(__name__)


class BaseQueryEngine(ABC):
    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._query(str_or_query_bundle)

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._aquery(str_or_query_bundle)

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
