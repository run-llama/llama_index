import os
from typing import Any, Callable, Dict, List, Literal, Optional, Type

import numpy as np
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
    RetrievalMetricResult,
)

_AGG_FUNC: Dict[str, Callable] = {"mean": np.mean, "median": np.median, "max": np.max}


class HitRate(BaseRetrievalMetric):
    """Hit rate metric: Compute the proportion of matches between retrieved documents and expected documents."""

    metric_name: str = "hit_rate"

    def compute(
        self,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
    ) -> RetrievalMetricResult:
        """Compute metric."""
        if (
            retrieved_ids is None
            or expected_ids is None
            or not retrieved_ids
            or not expected_ids
        ):
            raise ValueError("Both retrieved ids and expected ids must be provided")

        expected_set = set(expected_ids)
        hits = sum(1 for doc_id in retrieved_ids if doc_id in expected_set)
        score = hits / len(expected_ids) if expected_ids else 0.0

        return RetrievalMetricResult(score=score)


class RR(BaseRetrievalMetric):
    """Reciprocal Rank (RR): Calculates the reciprocal rank of the first, and only the first, relevant retrieved document.
    returns 0 if no relevant retrieved docs are found.
    """

    metric_name: str = "rr"

    def compute(
        self,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
    ) -> RetrievalMetricResult:
        """Compute metric."""
        if (
            retrieved_ids is None
            or expected_ids is None
            or not retrieved_ids
            or not expected_ids
        ):
            raise ValueError("Both retrieved ids and expected ids must be provided")
        for i, id in enumerate(retrieved_ids):
            if id in expected_ids:
                return RetrievalMetricResult(
                    score=1.0 / (i + 1),
                )
        return RetrievalMetricResult(
            score=0.0,
        )


class MRR(BaseRetrievalMetric):
    """Mean Reciprocal Rank (MRR): Sums up the reciprocal rank score for each relevant retrieved document.
    Then divides by the count of relevant documents.
    """

    metric_name: str = "mrr"

    def compute(
        self,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
    ) -> RetrievalMetricResult:
        """Compute the Mean Reciprocal Rank given expected document IDs and retrieved document IDs."""
        if (
            retrieved_ids is None
            or expected_ids is None
            or not retrieved_ids
            or not expected_ids
        ):
            raise ValueError("Both retrieved ids and expected ids must be provided")

        expected_set = set(expected_ids)
        reciprocal_rank_sum = 0.0
        relevant_docs_count = 0

        for index, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_set:
                relevant_docs_count += 1
                reciprocal_rank_sum += 1.0 / (index + 1)

        if relevant_docs_count > 0:
            mrr_score = reciprocal_rank_sum / relevant_docs_count
            return RetrievalMetricResult(score=mrr_score)
        else:
            return RetrievalMetricResult(score=0.0)


class CohereRerankRelevancyMetric(BaseRetrievalMetric):
    """Cohere rerank relevancy metric."""

    model: str = Field(description="Cohere model name.")
    metric_name: str = "cohere_rerank_relevancy"

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "rerank-english-v2.0",
        api_key: Optional[str] = None,
    ):
        try:
            api_key = api_key or os.environ["COHERE_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in cohere api key or "
                "specify via COHERE_API_KEY environment variable "
            )
        try:
            from cohere import Client  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "Cannot import cohere package, please `pip install cohere`."
            )

        self._client = Client(api_key=api_key)
        super().__init__(model=model)

    def _get_agg_func(self, agg: Literal["max", "median", "mean"]) -> Callable:
        """Get agg func."""
        return _AGG_FUNC[agg]

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
        agg: Literal["max", "median", "mean"] = "max",
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        """Compute metric."""
        del expected_texts  # unused

        if retrieved_texts is None:
            raise ValueError("Retrieved texts must be provided")

        results = self._client.rerank(
            model=self.model,
            top_n=len(
                retrieved_texts
            ),  # i.e. get a rank score for each retrieved chunk
            query=query,
            documents=retrieved_texts,
        )
        relevance_scores = [r.relevance_score for r in results.results]
        agg_func = self._get_agg_func(agg)

        return RetrievalMetricResult(
            score=agg_func(relevance_scores), metadata={"agg": agg}
        )


METRIC_REGISTRY: Dict[str, Type[BaseRetrievalMetric]] = {
    "hit_rate": HitRate,
    "rr": RR,
    "mrr": MRR,
    "cohere_rerank_relevancy": CohereRerankRelevancyMetric,
}


def resolve_metrics(metrics: List[str]) -> List[Type[BaseRetrievalMetric]]:
    """Resolve metrics from list of metric names."""
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Invalid metric name: {metric}")

    return [METRIC_REGISTRY[metric] for metric in metrics]
