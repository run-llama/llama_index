import logging
from typing import Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import (
    AsyncStreamingResponse,
    RESPONSE_TYPE,
    Response,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.settings import Settings

logger = logging.getLogger(__name__)


class RetrySourceQueryEngine(BaseQueryEngine):
    """Retry with different source nodes."""

    def __init__(
        self,
        query_engine: RetrieverQueryEngine,
        evaluator: BaseEvaluator,
        llm: Optional[LLM] = None,
        max_retries: int = 3,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Run a BaseQueryEngine with retries."""
        self._query_engine = query_engine
        self._evaluator = evaluator
        self._llm = llm or Settings.llm
        self.max_retries = max_retries
        super().__init__(callback_manager=callback_manager or Settings.callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {"query_engine": self._query_engine, "evaluator": self._evaluator}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
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
            # Test source nodes
            source_evals = [
                self._evaluator.evaluate(
                    query=query_str,
                    response=typed_response.response,
                    contexts=[source_node.get_content()],
                )
                for source_node in typed_response.source_nodes
            ]
            orig_nodes = typed_response.source_nodes
            assert len(source_evals) == len(orig_nodes)
            new_docs = []
            for node, eval_result in zip(orig_nodes, source_evals):
                if eval_result:
                    new_docs.append(Document(text=node.node.get_content()))
            if len(new_docs) == 0:
                raise ValueError("No source nodes passed evaluation.")
            new_index = SummaryIndex.from_documents(
                new_docs,
            )
            new_retriever_engine = RetrieverQueryEngine(new_index.as_retriever())
            new_query_engine = RetrySourceQueryEngine(
                new_retriever_engine,
                self._evaluator,
                self._llm,
                self.max_retries - 1,
            )
            return new_query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)
