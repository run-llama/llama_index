import logging
from typing import Optional, Sequence
from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.query_engine.types import QueryEngineWithMetadata
from gpt_index.response.schema import RESPONSE_TYPE
from gpt_index.selectors.llm_selectors import LLMSingleSelector
from gpt_index.selectors.types import BaseSelector

logger = logging.getLogger(__name__)


class RouterQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        selector: BaseSelector,
        query_engines_with_metadata: Sequence[QueryEngineWithMetadata],
    ) -> None:
        self._selector = selector
        self._query_engines = [x.query_engine for x in query_engines_with_metadata]
        self._metadatas = [x.metadata for x in query_engines_with_metadata]

    @classmethod
    def from_defaults(
        cls,
        query_engines_with_metadata: Sequence[QueryEngineWithMetadata],
        selector: Optional[BaseSelector] = None,
    ):
        selector = selector or LLMSingleSelector()
        return cls(selector, query_engines_with_metadata)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        result = self._selector.select(self._metadatas, query_bundle)
        try:
            selected_query_engine = self._query_engines[result.ind]
            logger.info(f"Selecting query engine {result.ind}.")
        except ValueError as e:
            raise ValueError("Failed to select query engine") from e

        return selected_query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        result = await self._selector.aselect(self._metadatas, query_bundle)
        try:
            selected_query_engine = self._query_engines[result.ind]
            logger.info(f"Selecting query engine {result.ind}.")
        except ValueError as e:
            raise ValueError("Failed to select query engine") from e

        return await selected_query_engine.aquery(query_bundle)
