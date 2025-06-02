import logging
from typing import Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    Response,
    AsyncStreamingResponse,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.evaluation.base import BaseEvaluator
from llama_index.core.evaluation.guideline import GuidelineEvaluator
from llama_index.core.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)


class RetryQueryEngine(BaseQueryEngine):
    """
    Does retry on query engine if it fails evaluation.

    Args:
        query_engine (BaseQueryEngine): A query engine object
        evaluator (BaseEvaluator): An evaluator object
        max_retries (int): Maximum number of retries
        callback_manager (Optional[CallbackManager]): A callback manager object

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

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {"query_engine": self._query_engine, "evaluator": self._evaluator}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        assert not isinstance(response, AsyncStreamingResponse)
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
            query_transformer = FeedbackQueryTransformation()
            new_query = query_transformer.run(query_bundle, {"evaluation": eval})
            return new_query_engine.query(new_query)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)


class RetryGuidelineQueryEngine(BaseQueryEngine):
    """
    Does retry with evaluator feedback
    if query engine fails evaluation.

    Args:
        query_engine (BaseQueryEngine): A query engine object
        guideline_evaluator (GuidelineEvaluator): A guideline evaluator object
        resynthesize_query (bool): Whether to resynthesize query
        max_retries (int): Maximum number of retries
        callback_manager (Optional[CallbackManager]): A callback manager object

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        guideline_evaluator: GuidelineEvaluator,
        resynthesize_query: bool = False,
        max_retries: int = 3,
        callback_manager: Optional[CallbackManager] = None,
        query_transformer: Optional[FeedbackQueryTransformation] = None,
    ) -> None:
        self._query_engine = query_engine
        self._guideline_evaluator = guideline_evaluator
        self.max_retries = max_retries
        self.resynthesize_query = resynthesize_query
        self.query_transformer = query_transformer or FeedbackQueryTransformation(
            resynthesize_query=self.resynthesize_query
        )
        super().__init__(callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "query_engine": self._query_engine,
            "guideline_evalator": self._guideline_evaluator,
        }

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        assert not isinstance(response, AsyncStreamingResponse)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._guideline_evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryGuidelineQueryEngine(
                self._query_engine,
                self._guideline_evaluator,
                self.resynthesize_query,
                self.max_retries - 1,
                self.callback_manager,
            )
            new_query = self.query_transformer.run(query_bundle, {"evaluation": eval})
            logger.debug("New query: %s", new_query.query_str)
            return new_query_engine.query(new_query)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)
