import logging

from typing import Optional
from llama_index.callbacks.base import CallbackManager
from llama_index.evaluation.base import BaseEvaluator
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE, Response

logger = logging.getLogger(__name__)


class RetryQueryEngine(BaseQueryEngine):
    """Retriever query engine with retry.

    Args:
        base_query_engine (BaseQueryEngine): A query engine object

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        evaluator: BaseEvaluator,
        max_retries: int = 3,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._query_engine = query_engine
        self._evaluator = evaluator
        self.max_retries = max_retries
        super().__init__(callback_manager)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryQueryEngine(
                self._query_engine, self._evaluator, self.max_retries - 1
            )
            query_transformer = FeedbackQueryTransformation(eval)
            new_query = query_transformer.run(query_bundle)
            return new_query_engine.query(new_query)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)
