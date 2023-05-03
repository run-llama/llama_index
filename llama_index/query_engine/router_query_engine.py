import logging
from typing import Optional, Sequence
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.selectors.types import BaseSelector
from llama_index.tools.query_engine import QueryEngineTool

logger = logging.getLogger(__name__)


class RouterQueryEngine(BaseQueryEngine):
    """Router query engine.

    Selects one out of several candidate query engines to execute a query.

    Args:
        selector (BaseSelector): A selector that chooses one out of many options based
            on each candidate's metadata and query.
        query_engine_tools (Sequence[QueryEngineTool]): A sequence of candidate
            query engines. They must be wrapped as tools to expose metadata to
            the selector.

    """

    def __init__(
        self,
        selector: BaseSelector,
        query_engine_tools: Sequence[QueryEngineTool],
    ) -> None:
        self._selector = selector
        self._query_engines = [x.query_engine for x in query_engine_tools]
        self._metadatas = [x.metadata for x in query_engine_tools]

    @classmethod
    def from_defaults(
        cls,
        query_engine_tools: Sequence[QueryEngineTool],
        selector: Optional[BaseSelector] = None,
    ) -> "RouterQueryEngine":
        selector = selector or LLMSingleSelector.from_defaults()
        return cls(selector, query_engine_tools)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        result = self._selector.select(self._metadatas, query_bundle)
        try:
            selected_query_engine = self._query_engines[result.ind]
            logger.info(f"Selecting query engine {result.ind}: {result.reason}.")
        except ValueError as e:
            raise ValueError("Failed to select query engine") from e

        return selected_query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        result = await self._selector.aselect(self._metadatas, query_bundle)
        try:
            selected_query_engine = self._query_engines[result.ind]
            logger.info(f"Selecting query engine {result.ind}: {result.reason}.")
        except ValueError as e:
            raise ValueError("Failed to select query engine") from e

        return await selected_query_engine.aquery(query_bundle)
