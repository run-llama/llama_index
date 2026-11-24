from types import SimpleNamespace
from unittest.mock import Mock
from typing import Any, Optional, Sequence

import pytest

from llama_index.core.base.response.schema import NodeWithScore, Response
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.retrieval.base import BaseRetrievalEvaluator
from llama_index.core.evaluation.retrieval.metrics import (
    CohereRerankRelevancyMetric,
    HitRate,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import TextNode


class MockEvaluator(BaseEvaluator):
    def __init__(
        self,
        mock_score: float = 1.0,
        mock_passing: bool = True,
        mock_feedback: str = "test feedback",
    ) -> None:
        self._mock_score = mock_score
        self._mock_passing = mock_passing
        self._mock_feedback = mock_feedback

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        return EvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            passing=self._mock_passing,
            score=self._mock_score,
            feedback=self._mock_feedback,
        )


def test_evaluator_basic() -> None:
    test_evaluator = MockEvaluator()
    eval_result_0 = test_evaluator.evaluate(
        query="test query",
        response="test response",
        contexts=["test context 1", "test context 2"],
    )

    eval_result_1 = test_evaluator.evaluate_response(
        query="test query",
        response=Response(
            response="test response",
            source_nodes=[
                NodeWithScore(node=TextNode(text="test context 1"), score=1.0),
                NodeWithScore(node=TextNode(text="test context 2"), score=1.0),
            ],
        ),
    )

    assert eval_result_0 == eval_result_1


class MockRetrievalEvaluator(BaseRetrievalEvaluator):
    async def _aget_retrieved_ids_and_texts(self, query, mode):
        return ["a", "b", "c"], ["first", "second", "third"]


@pytest.fixture()
def rerank_evaluator():
    # Bypass SDK initialization so scoring is tested without the optional SDK.
    metric = CohereRerankRelevancyMetric.model_construct(model="mock-reranker")
    metric._client = SimpleNamespace(
        rerank=Mock(
            return_value=SimpleNamespace(
                results=[
                    SimpleNamespace(relevance_score=score) for score in [0.1, 0.3, 0.8]
                ]
            )
        )
    )
    return MockRetrievalEvaluator(metrics=[metric, HitRate()])


@pytest.mark.parametrize(
    ("kwargs", "expected_score"),
    [
        ({}, 0.8),
        ({"agg": "max"}, 0.8),
        ({"agg": "mean"}, 0.4),
        ({"agg": "median"}, 0.3),
    ],
)
def test_retrieval_evaluate_metric_kwargs(rerank_evaluator, kwargs, expected_score):
    result = rerank_evaluator.evaluate("query", ["a"], **kwargs)

    assert result.metric_dict["hit_rate"].score == 1.0
    metric_result = result.metric_dict["cohere_rerank_relevancy"]
    assert metric_result.score == pytest.approx(expected_score)
    assert metric_result.metadata == {"agg": kwargs.get("agg", "max")}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected_score"),
    [
        ({}, 0.8),
        ({"agg": "max"}, 0.8),
        ({"agg": "mean"}, 0.4),
        ({"agg": "median"}, 0.3),
    ],
)
async def test_retrieval_aevaluate_metric_kwargs(
    rerank_evaluator, kwargs, expected_score
):
    result = await rerank_evaluator.aevaluate("query", ["a"], **kwargs)

    assert result.metric_dict["hit_rate"].score == 1.0
    metric_result = result.metric_dict["cohere_rerank_relevancy"]
    assert metric_result.score == pytest.approx(expected_score)
    assert metric_result.metadata == {"agg": kwargs.get("agg", "max")}
