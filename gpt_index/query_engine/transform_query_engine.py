from typing import Optional
from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.query_transform.base import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.response.schema import RESPONSE_TYPE


class TransformQueryEngine(BaseQueryEngine):
    """Transform query engine.

    Applies a query transform to a query bundle before passing
        it to a query engine.

    Args:
        query_engine (BaseQueryEngine): A query engine object.
        query_transform (BaseQueryTransform): A query transform object.
        transform_extra_info (Optional[dict]): Extra info to pass to the
            query transform.

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        query_transform: BaseQueryTransform,
        transform_extra_info: Optional[dict] = None,
    ) -> None:
        self._query_engine = query_engine
        self._query_transform = query_transform
        self._transform_extra_info = transform_extra_info

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        query_bundle = self._query_transform.run(
            query_bundle, extra_info=self._transform_extra_info
        )
        return self._query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        query_bundle = self._query_transform.run(
            query_bundle, extra_info=self._transform_extra_info
        )
        return await self._query_engine.aquery(query_bundle)
