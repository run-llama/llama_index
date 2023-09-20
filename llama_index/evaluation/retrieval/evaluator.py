"""Retrieval evaluators."""

from pydantic import Field
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, List, Dict

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.response.schema import Response
from llama_index.evaluation.retrieval.metrics_base import BaseRetrievalMetric, RetrievalMetricResult
from llama_index.evaluation.retrieval.metrics import resolve_metrics
from llama_index.evaluation.retrieval.base import BaseRetrievalEvaluator, RetrievalEvalResult
from llama_index.indices.base_retriever import BaseRetriever



class RetrieverEvaluator(BaseRetrievalEvaluator):
    """Retriever evaluator.

    Args:
        metrics (List[BaseRetrievalMetric]): Sequence of metrics to evaluate
        retriever: Retriever to evaluate.
    
    """

    retriever: BaseRetriever = Field(
        ..., description="Retriever to evaluate"
    )

    def __init__(
        self,
        metrics: Sequence[BaseRetrievalMetric],
        retriever: BaseRetriever,
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(metrics=metrics, retriever=retriever, **kwargs)

    async def aevaluate(
        self,
        query: str,
        expected_ids: List[str],
        **kwargs: Any
    ) -> RetrievalEvalResult:
        """Evaluate retriever.

        Args:
            query (str): Query string
            expected_ids (List[str]): Expected ids
        
        """
        retrieved_nodes = await self.retriever.aretrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        metric_dict = {}
        for metric in self.metrics:
            eval_result = metric.compute(query, expected_ids, retrieved_ids)
            metric_dict[metric.metric_name] = eval_result
        return RetrievalEvalResult(
            query=query,
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            metric_dict=metric_dict,
        )
            
    