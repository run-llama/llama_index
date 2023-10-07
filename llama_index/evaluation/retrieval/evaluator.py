"""Retrieval evaluators."""

from typing import Any, List, Sequence

from llama_index.bridge.pydantic import Field
from llama_index.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
)
from llama_index.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
)
from llama_index.indices.base_retriever import BaseRetriever


class RetrieverEvaluator(BaseRetrievalEvaluator):
    """Retriever evaluator.

    This module will evaluate a retriever using a set of metrics.

    Args:
        metrics (List[BaseRetrievalMetric]): Sequence of metrics to evaluate
        retriever: Retriever to evaluate.

    """

    retriever: BaseRetriever = Field(..., description="Retriever to evaluate")

    def __init__(
        self,
        metrics: Sequence[BaseRetrievalMetric],
        retriever: BaseRetriever,
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(metrics=metrics, retriever=retriever, **kwargs)

    async def _aget_retrieved_ids(self, query: str) -> List[str]:
        """Get retrieved ids."""
        retrieved_nodes = await self.retriever.aretrieve(query)
        return [node.node.node_id for node in retrieved_nodes]
