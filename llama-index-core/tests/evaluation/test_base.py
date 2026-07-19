from typing import Any, Optional, Sequence

import pytest

from llama_index.core.base.response.schema import NodeWithScore, Response
from llama_index.core.evaluation import BaseEvaluator
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


class _NonAsyncEvaluator(BaseEvaluator):
    """
    Evaluator that delegates ``aevaluate`` to the base implementation.

    ``BaseEvaluator.aevaluate`` is ``@abstractmethod`` with a
    ``raise NotImplementedError`` body. Subclasses normally override it, but
    here we deliberately call ``super().aevaluate(...)`` to verify the body
    raises a descriptive error rather than a bare one.
    """

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def evaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        return EvaluationResult(
            query=query, contexts=contexts, response=response, passing=True
        )

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        return await super().aevaluate(
            query=query, response=response, contexts=contexts, **kwargs
        )


@pytest.mark.asyncio
async def test_base_aevaluate_raises_descriptive() -> None:
    """BaseEvaluator.aevaluate must raise NotImplementedError with a helpful message."""
    evaluator = _NonAsyncEvaluator()
    with pytest.raises(NotImplementedError, match="aevaluate"):
        await evaluator.aevaluate(query="q", response="r", contexts=["c"])
