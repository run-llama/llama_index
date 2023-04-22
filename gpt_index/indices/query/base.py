"""Base query engine."""

import logging
from abc import ABC, abstractmethod
from typing import Union

from gpt_index.indices.query.schema import QueryBundle
from gpt_index.response.schema import (
    RESPONSE_TYPE,
)

logger = logging.getLogger(__name__)


class BaseQueryEngine(ABC):
    def query(self, str_or_query_bundle: Union[str, QueryBundle]) -> RESPONSE_TYPE:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._query(str_or_query_bundle)

    async def aquery(
        self, str_or_query_bundle: Union[str, QueryBundle]
    ) -> RESPONSE_TYPE:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._aquery(str_or_query_bundle)

    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass
