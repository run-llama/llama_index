from typing import Any, Optional, Sequence

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


def test_evaluate_response_with_response_field() -> None:
    """Test that response_field extracts value from response.metadata."""
    test_evaluator = MockEvaluator()
    large_response = "x" * 400_000  # simulate huge SQL result set
    sql_query = "SELECT * FROM users LIMIT 10"

    result = test_evaluator.evaluate_response(
        query="test query",
        response=Response(
            response=large_response,
            source_nodes=[
                NodeWithScore(node=TextNode(text="ctx"), score=1.0),
            ],
            metadata={"sql_query": sql_query},
        ),
        response_field="sql_query",
    )

    # The evaluator should receive the compact sql_query, not the huge response
    assert result.response == sql_query


def test_evaluate_response_field_missing_key() -> None:
    """Test that a missing metadata key yields None response string."""
    test_evaluator = MockEvaluator()

    result = test_evaluator.evaluate_response(
        query="test query",
        response=Response(
            response="some response",
            source_nodes=[],
            metadata={"other_key": "value"},
        ),
        response_field="nonexistent_key",
    )

    assert result.response is None


def test_evaluate_response_field_none_metadata() -> None:
    """Test that None metadata with response_field yields None response string."""
    test_evaluator = MockEvaluator()

    result = test_evaluator.evaluate_response(
        query="test query",
        response=Response(
            response="some response",
            source_nodes=[],
            metadata=None,
        ),
        response_field="sql_query",
    )

    assert result.response is None


def test_evaluate_response_without_response_field_unchanged() -> None:
    """Test that default behavior (no response_field) is unchanged."""
    test_evaluator = MockEvaluator()

    result = test_evaluator.evaluate_response(
        query="test query",
        response=Response(
            response="original response",
            source_nodes=[],
            metadata={"sql_query": "SELECT 1"},
        ),
    )

    assert result.response == "original response"
