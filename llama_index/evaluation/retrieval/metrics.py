from typing import Any, Dict, List, Optional

from llama_index.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
    RetrievalMetricResult,
)


class HitRate(BaseRetrievalMetric):
    """Hit rate metric."""

    metric_name: str = "hit_rate"

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        """Compute metric."""
        if retrieved_ids is None or expected_ids is None:
            raise ValueError("Retrieved ids and expected ids must be provided")
        is_hit = any(id in expected_ids for id in retrieved_ids)
        return RetrievalMetricResult(
            score=1.0 if is_hit else 0.0,
        )


class MRR(BaseRetrievalMetric):
    """MRR metric."""

    metric_name: str = "mrr"

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        """Compute metric."""
        if retrieved_ids is None or expected_ids is None:
            raise ValueError("Retrieved ids and expected ids must be provided")
        for i, id in enumerate(retrieved_ids):
            if id in expected_ids:
                return RetrievalMetricResult(
                    score=1.0 / (i + 1),
                )
        return RetrievalMetricResult(
            score=0.0,
        )


METRIC_REGISTRY: Dict[str, BaseRetrievalMetric] = {
    "hit_rate": HitRate(),
    "mrr": MRR(),
}


def resolve_metrics(metrics: List[str]) -> List[BaseRetrievalMetric]:
    """Resolve metrics from list of metric names."""
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Invalid metric name: {metric}")

    return [METRIC_REGISTRY[metric] for metric in metrics]
