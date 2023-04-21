"""Base query engine."""

import logging
from abc import ABC, abstractmethod

from gpt_index.indices.query.schema import QueryBundle
from gpt_index.response.schema import (
    RESPONSE_TYPE,
)

logger = logging.getLogger(__name__)


class BaseQueryEngine(ABC):
    @abstractmethod
    def query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass
