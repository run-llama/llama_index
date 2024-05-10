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
    """Hit rate metric: Compute hit rate with two calculation options.

    - The default method checks for a single match between any of the retrieved docs and expected docs.
    - The more granular method checks for all potential matches between retrieved docs and expected docs.

    Attributes:
        use_granular_hit_rate (bool): Determines whether to use the granular method for calculation.
        metric_name (str): The name of the metric.
    """

    metric_name: str = "hit_rate"
    use_granular_hit_rate: bool = False

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
    ) -> RetrievalMetricResult:
        """Compute metric based on the provided inputs.

        Parameters:
            query (Optional[str]): The query string (not used in the current implementation).
            expected_ids (Optional[List[str]]): Expected document IDs.
            retrieved_ids (Optional[List[str]]): Retrieved document IDs.
            expected_texts (Optional[List[str]]): Expected texts (not used in the current implementation).
            retrieved_texts (Optional[List[str]]): Retrieved texts (not used in the current implementation).

        Raises:
            ValueError: If the necessary IDs are not provided.

        Returns:
            RetrievalMetricResult: The result with the computed hit rate score.
        """
        # Checking for the required arguments
        if (
            retrieved_ids is None
            or expected_ids is None
            or not retrieved_ids
            or not expected_ids
        ):
            raise ValueError("Retrieved ids and expected ids must be provided")

        if self.use_granular_hit_rate:
            # Granular HitRate calculation: Calculate all hits and divide by the number of expected docs
            expected_set = set(expected_ids)
            hits = sum(1 for doc_id in retrieved_ids if doc_id in expected_set)
            score = hits / len(expected_ids) if expected_ids else 0.0
        else:
            # Default HitRate calculation: Check if there is a single hit
            is_hit = any(id in expected_ids for id in retrieved_ids)
            score = 1.0 if is_hit else 0.0

        return RetrievalMetricResult(score=score)


class MRR(BaseRetrievalMetric):
    """MRR (Mean Reciprocal Rank) metric with two calculation options.

    - The default method calculates the reciprocal rank of the first relevant retrieved document.
    - The more granular method sums the reciprocal ranks of all relevant retrieved documents and divides by the count of relevant documents.

    Attributes:
        use_granular_mrr (bool): Determines whether to use the granular method for calculation.
        metric_name (str): The name of the metric.
    """

    metric_name: str = "mrr"
    use_granular_mrr: bool = False

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
    ) -> RetrievalMetricResult:
        """Compute MRR based on the provided inputs and selected method.

        Parameters:
            query (Optional[str]): The query string (not used in the current implementation).
            expected_ids (Optional[List[str]]): Expected document IDs.
            retrieved_ids (Optional[List[str]]): Retrieved document IDs.
            expected_texts (Optional[List[str]]): Expected texts (not used in the current implementation).
            retrieved_texts (Optional[List[str]]): Retrieved texts (not used in the current implementation).

        Raises:
            ValueError: If the necessary IDs are not provided.

        Returns:
            RetrievalMetricResult: The result with the computed MRR score.
        """
        # Checking for the required arguments
        if (
            retrieved_ids is None
            or expected_ids is None
            or not retrieved_ids
            or not expected_ids
        ):
            raise ValueError("Retrieved ids and expected ids must be provided")

        if self.use_granular_mrr:
            # Granular MRR calculation: All relevant retrieved docs have their reciprocal ranks summed and averaged
            expected_set = set(expected_ids)
            reciprocal_rank_sum = 0.0
            relevant_docs_count = 0
            for index, doc_id in enumerate(retrieved_ids):
                if doc_id in expected_set:
                    relevant_docs_count += 1
                    reciprocal_rank_sum += 1.0 / (index + 1)
            mrr_score = (
                reciprocal_rank_sum / relevant_docs_count
                if relevant_docs_count > 0
                else 0.0
            )
        else:
            # Default MRR calculation: Reciprocal rank of the first relevant document retrieved
            for i, id in enumerate(retrieved_ids):
                if id in expected_ids:
                    return RetrievalMetricResult(score=1.0 / (i + 1))
            mrr_score = 0.0

        return RetrievalMetricResult(score=mrr_score)


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
    "mrr": MRR,
    "cohere_rerank_relevancy": CohereRerankRelevancyMetric,
}


def resolve_metrics(metrics: List[str]) -> List[Type[BaseRetrievalMetric]]:
    """Resolve metrics from list of metric names."""
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Invalid metric name: {metric}")

    return [METRIC_REGISTRY[metric] for metric in metrics]
