import os
from typing import Any, List, Literal, Optional

import numpy as np

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.evaluation.retrieval.metrics_base import (
    BaseIndexlessRetrievalMetric,
    RetrievalMetricResult,
)


class CohereRerankRelevancyMetric(BaseIndexlessRetrievalMetric):
    """Cohere rerank relevancy metric."""

    metric_name: str = "hit_rate"

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
            from cohere import Client
        except ImportError:
            raise ImportError(
                "Cannot import cohere package, please `pip install cohere`."
            )

        self._client = Client(api_key=api_key)
        super().__init__(model=model)

    def compute(
        self,
        query: Optional[str] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
        agg: Literal["max", "median", "mean"] = "max",
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        """Compute metric."""
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

        match agg:
            case "max":
                score = np.max(r.relevance_score for r in results)
            case "median":
                score = np.median(r.relevance_score for r in results)
            case "mean":
                score = np.mean(r.relevance_score for r in results)

        return RetrievalMetricResult(score=score, metadata={"agg": agg})
