import asyncio
from typing import Any, List, Optional, Sequence

import pytest
from llama_index.core.base.response.schema import NodeWithScore, Response
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.evaluation import BaseEvaluator, SemanticSimilarityEvaluator
from llama_index.core.evaluation.base import EvaluationResult
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


@pytest.mark.asyncio
async def test_semantic_similarity_evaluator_concurrent_and_metadata() -> None:
    in_flight = 0
    max_in_flight = 0

    class ConcurrentTrackingEmbedModel(MockEmbedding):
        async def _aget_text_embedding(self, text: str) -> List[float]:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return [1.0, 0.0] if text == "resp" else [1.0, 0.0]

    evaluator = SemanticSimilarityEvaluator(
        embed_model=ConcurrentTrackingEmbedModel(embed_dim=2),
        similarity_threshold=0.8,
    )
    result = await evaluator.aevaluate(
        query="q",
        response="resp",
        contexts=["ctx"],
        reference="ref",
    )
    assert max_in_flight == 2
    assert result.query == "q"
    assert result.response == "resp"
    assert result.contexts == ["ctx"]
    assert result.passing is True
    assert result.score == pytest.approx(1.0)
