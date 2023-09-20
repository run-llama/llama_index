"""Base retrieval abstractions."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, List, Dict
from pydantic import Field

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.response.schema import Response
from llama_index.evaluation.retrieval.metrics_base import BaseRetrievalMetric, RetrievalMetricResult
from llama_index.evaluation.retrieval.metrics import resolve_metrics


class RetrievalEvalResult(BaseModel):
    """Retrieval eval result.
    
    Attributes:
        query (str): Query string
        expected_ids (List[str]): Expected ids
        retrieved_ids (List[str]): Retrieved ids
        metric_dict (Dict[str, BaseRetrievalMetric]): Metric dictionary for the evaluation
    
    """

    class Config:
        arbitrary_types_allowed = True

    query: str = Field(..., description="Query string")
    expected_ids: List[str] = Field(..., description="Expected ids")
    retrieved_ids: List[str] = Field(..., description="Retrieved ids")

    metric_dict: Dict[str, RetrievalMetricResult] = Field(
        ..., description="Metric dictionary for the evaluation"
    )

    @property
    def metric_vals_dict(self) -> Dict[str, float]:
        """Dictionary of metric values."""
        return {k: v.score for k, v in self.metric_dict.items()}


class BaseRetrievalEvaluator(BaseModel):
    """Base Retrieval Evaluator class."""

    metrics: Sequence[BaseRetrievalMetric] = Field(
        ..., description="Sequence of metrics to evaluate"
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_metric_names(
        cls,
        metric_names: Sequence[str],
        **kwargs: Any
    ) -> "BaseRetrievalEvaluator":
        """Create evaluator from metric names.

        Args:
            metric_names (Sequence[str]): Sequence of metric names
            **kwargs: Additional arguments for the evaluator
        
        """
        metrics = resolve_metrics(metric_names)
        return cls(metrics=metrics, **kwargs)

    def evaluate(
        self,
        query: str,
        expected_ids: List[str],
        **kwargs: Any
    ) -> RetrievalEvalResult:
        """Run evaluation results with query string and expected ids.
        
        Args:
            query (str): Query string
            expected_ids (List[str]): Expected ids

        Returns:
            RetrievalEvalResult: Evaluation result
        
        """
        return asyncio.run(
            self.aevaluate(query=query, expected_ids=expected_ids, **kwargs)
        )

    @abstractmethod
    async def aevaluate(
        self,
        query: str,
        expected_ids: List[str],
        **kwargs: Any,
    ) -> RetrievalEvalResult:
        """Run evaluation with query string, retrieved contexts,
        and generated response string.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        raise NotImplementedError