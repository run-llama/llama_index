from typing import Sequence
from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.query_engine.types import QueryEngineWithMetadata
from gpt_index.response.schema import RESPONSE_TYPE
from gpt_index.selectors.types import BaseSelector


class RouterQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        selector: BaseSelector,
        query_engines_with_metadata: Sequence[QueryEngineWithMetadata],
    ) -> None:
        self._selector = selector
        self._query_engines = [x.query_engine for x in query_engines_with_metadata]
        self._metadatas = [x.metadata for x in query_engines_with_metadata]

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        result = self._selector.select(self._metadatas, query_bundle)
        try:
            ind = result.selection_ind
            selected_query_engine = self._query_engines[ind]
        except ValueError as e:
            raise ValueError("Failed to select query engine") from e

        return selected_query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        result = await self._selector.aselect(self._metadatas, query_bundle)
        try:
            ind = result.selection_ind
            selected_query_engine = self._query_engines[ind]
        except ValueError as e:
            raise ValueError("Failed to select query engine") from e

        return await selected_query_engine.aquery(query_bundle)
