import logging

from typing import Optional
from llama_index.callbacks.base import CallbackManager
from llama_index.evaluation.base import QueryResponseEvaluator
from llama_index.indices.list.base import GPTListIndex
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.readers.schema.base import Document
from llama_index.response.schema import RESPONSE_TYPE, Response

logger = logging.getLogger(__name__)


class RetrySourceQueryEngine(BaseQueryEngine):
    """Retry with different source nodes."""

    def __init__(
        self,
        query_engine: RetrieverQueryEngine,
        evaluator: QueryResponseEvaluator,
        max_retries: int = 3,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Run a BaseQueryEngine with retries."""
        self._query_engine = query_engine
        self._evaluator = evaluator
        self.max_retries = max_retries
        super().__init__(callback_manager)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
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
            # Test source nodes
            source_evals = self._evaluator.evaluate_source_nodes(
                query_str, typed_response
            )
            orig_nodes = typed_response.source_nodes
            assert len(source_evals) == len(orig_nodes)
            new_docs = []
            for node, eval_result in zip(orig_nodes, source_evals):
                if eval_result:
                    new_docs.append(Document(node.node.get_text()))
            if len(new_docs) == 0:
                raise ValueError("No source nodes passed evaluation.")
            new_index = GPTListIndex.from_documents(new_docs)
            new_retriever_engine = RetrieverQueryEngine(new_index.as_retriever())
            new_query_engine = RetrySourceQueryEngine(
                new_retriever_engine, self._evaluator, self.max_retries - 1
            )
            return new_query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)
