from typing import Sequence
from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.response.schema import RESPONSE_TYPE
from gpt_index.selectors.types import BaseSelector, Metadata


class RouterQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        selector: BaseSelector,
        query_engines: Sequence[BaseQueryEngine],
        metadatas: Sequence[Metadata],
    ) -> None:
        self._selector = selector
        self._query_engines = query_engines
        self._metadatas = metadatas

        if len(self._query_engines) != len(self._metadatas):
            raise ValueError("Length of query engines and metadatas must match.")

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
