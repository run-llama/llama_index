import math
import os
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Type

import numpy as np
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
    RetrievalMetricResult,
)
from typing_extensions import assert_never

_AGG_FUNC: Dict[str, Callable] = {"mean": np.mean, "median": np.median, "max": np.max}


class HitRate(BaseRetrievalMetric):
    metric_name: ClassVar[str] = "hit_rate"
    use_granular_hit_rate: bool = False

    def compute(self, query=None, expected_ids=None, retrieved_ids=None,
               expected_texts=None, retrieved_texts=None, **kwargs):
        if (retrieved_ids is None or expected_ids is None or not retrieved_ids
                or not expected_ids):
            raise ValueError("Retrieved ids and expected ids must be provided")
        if self.use_granular_hit_rate:
            expected_set = set(expected_ids)
            hits = sum(1 for doc_id in retrieved_ids if doc_id in expected_set)
            score = hits / len(expected_ids) if expected_ids else 0.0
        else:
            is_hit = any(id in expected_ids for id in retrieved_ids)
            score = 1.0 if is_hit else 0.0
        return RetrievalMetricResult(score=score)


class MRR(BaseRetrievalMetric):
    metric_name: ClassVar[str] = "mrr"
    use_granular_mrr: bool = False

    def compute(self, query=None, expected_ids=None, retrieved_ids=None,
               expected_texts=None, retrieved_texts=None, **kwargs):
        if (retrieved_ids is None or expected_ids is None or not retrieved_ids
                or not expected_ids):
            raise ValueError("Retrieved ids and expected ids must be provided")
        if self.use_granular_mrr:
            expected_set = set(expected_ids)
            reciprocal_rank_sum = 0.0
            relevant_docs_count = 0
            for index, doc_id in enumerate(retrieved_ids):
                if doc_id in expected_set:
                    relevant_docs_count += 1
                    reciprocal_rank_sum += 1.0 / (index + 1)
            mrr_score = (reciprocal_rank_sum / relevant_docs_count
                         if relevant_docs_count > 0 else 0.0)
        else:
            for i, id in enumerate(retrieved_ids):
                if id in expected_ids:
                    return RetrievalMetricResult(score=1.0 / (i + 1))
            mrr_score = 0.0
        return RetrievalMetricResult(score=mrr_score)


class Precision(BaseRetrievalMetric):
    metric_name: ClassVar[str] = "precision"

    def compute(self, query=None, expected_ids=None, retrieved_ids=None,
               expected_texts=None, retrieved_texts=None, **kwargs):
        if (retrieved_ids is None or expected_ids is None or not retrieved_ids
                or not expected_ids):
            raise ValueError("Retrieved ids and expected ids must be provided")
        retrieved_set = set(retrieved_ids)
        expected_set = set(expected_ids)
        precision = len(retrieved_set & expected_set) / len(retrieved_set)
        return RetrievalMetricResult(score=precision)


class Recall(BaseRetrievalMetric):
    metric_name: ClassVar[str] = "recall"

    def compute(self, query=None, expected_ids=None, retrieved_ids=None,
               expected_texts=None, retrieved_texts=None, **kwargs):
        if (retrieved_ids is None or expected_ids is None or not retrieved_ids
                or not expected_ids):
            raise ValueError("Retrieved ids and expected ids must be provided")
        retrieved_set = set(retrieved_ids)
        expected_set = set(expected_ids)
        recall = len(retrieved_set & expected_set) / len(expected_set)
        return RetrievalMetricResult(score=recall)


class AveragePrecision(BaseRetrievalMetric):
    metric_name: ClassVar[str] = "ap"

    def compute(self, query=None, expected_ids=None, retrieved_ids=None,
               expected_texts=None, retrieved_texts=None, **kwargs):
        if (retrieved_ids is None or expected_ids is None or not retrieved_ids
                or not expected_ids):
            raise ValueError("Retrieved ids and expected ids must be provided")
        expected_set = set(expected_ids)

        # Deduplicate retrieved_ids (preserving order) so that a downstream caller
        # that merges results from several retrievers cannot count the same relevant
        # document more than once, which would inflate average precision above 1.0.
        seen: set = set()
        retrieved_ids = [
            doc_id for doc_id in retrieved_ids
            if not (doc_id in seen or seen.add(doc_id))
        ]

        relevant_count, total_precision = 0, 0.0
        for i, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in expected_set:
                relevant_count += 1
                total_precision += relevant_count / i

        average_precision = total_precision / len(expected_set)
        return RetrievalMetricResult(score=average_precision)


DiscountedGainMode = Literal["linear", "exponential"]


def discounted_gain(*, rel: float, i: int, mode: DiscountedGainMode) -> float:
    if rel == 0:
        return 0
    if rel == 1:
        return 1 / math.log2(i + 1)
    if mode == "linear":
        return rel / math.log2(i + 1)
    elif mode == "exponential":
        return (2**rel - 1) / math.log2(i + 1)
    else:
        assert_never(mode)


class NDCG(BaseRetrievalMetric):
    metric_name: ClassVar[str] = "ndcg"
    mode: DiscountedGainMode = "linear"

    def compute(self, query=None, expected_ids=None, retrieved_ids=None,
               expected_texts=None, retrieved_texts=None, **kwargs):
        if (retrieved_ids is None or expected_ids is None or not retrieved_ids
                or not expected_ids):
            raise ValueError("Retrieved ids and expected ids must be provided")
        mode = self.mode
        expected_set = set(expected_ids)

        # Deduplicate retrieved_ids (preserving order): a duplicated relevant id
        # must not contribute gain more than once, which would otherwise inflate
        # NDCG above 1.0.
        seen: set = set()
        retrieved_ids = [
            doc_id for doc_id in retrieved_ids
            if not (doc_id in seen or seen.add(doc_id))
        ]

        dcg = sum(
            discounted_gain(rel=docid in expected_set, i=i, mode=mode)
            for i, docid in enumerate(retrieved_ids, start=1)
        )
        idcg = sum(
            discounted_gain(rel=True, i=i, mode=mode)
            for i in range(1, len(expected_ids) + 1)
        )
        ndcg_score = dcg / idcg
        return RetrievalMetricResult(score=ndcg_score)


class CohereRerankRelevancyMetric(BaseRetrievalMetric):
    metric_name: ClassVar[str] = "cohere_rerank_relevancy"
    model: str = Field(description="Cohere model name.")
    _client: Any = PrivateAttr()

    def __init__(self, model="rerank-english-v2.0", api_key=None):
        try:
            api_key = api_key or os.environ["COHERE_API_KEY"]
        except IndexError:
            raise ValueError("Must pass in cohere api key or COHERE_API_KEY")
        try:
            from cohere import Client
        except ImportError:
            raise ImportError("Cannot import cohere package, please `pip install cohere`.")
        super().__init__(model=model)
        self._client = Client(api_key=api_key)

    def _get_agg_func(self, agg):
        return _AGG_FUNC[agg]

    def compute(self, query=None, expected_ids=None, retrieved_ids=None,
               expected_texts=None, retrieved_texts=None, agg="max", **kwargs):
        del expected_texts
        if retrieved_texts is None:
            raise ValueError("Retrieved texts must be provided")
        results = self._client.rerank(model=model, top_n=len(retrieved_texts),
                                      query=query, documents=retrieved_texts)
        relevance_scores = [r.relevance_score for r in results.results]
        agg_func = self._get_agg_func(agg)
        return RetrievalMetricResult(score=agg_func(relevance_scores), metadata={"agg": agg})


METRIC_REGISTRY: Dict[str, Type[BaseRetrievalMetric]] = {
    "hit_rate": HitRate,
    "mrr": MRR,
    "precision": Precision,
    "recall": Recall,
    "ap": AveragePrecision,
    "ndcg": NDCG,
    "cohere_rerank_relevancy": CohereRerankRelevancyMetric,
}


def resolve_metrics(metrics):
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Invalid metric name: {metric}")
    return [METRIC_REGISTRY[metric] for metric in metrics]
