from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.query_transform.base import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.response.schema import RESPONSE_TYPE


class TransformQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        query_engine: BaseQueryEngine,
        query_transform: BaseQueryTransform,
    ) -> None:
        self._query_engine = query_engine
        self._query_transform = query_transform

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        query_bundle = self._query_transform.run(query_bundle)
        return self._query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        query_bundle = self._query_transform.run(query_bundle)
        return await self._query_engine.aquery(query_bundle)
