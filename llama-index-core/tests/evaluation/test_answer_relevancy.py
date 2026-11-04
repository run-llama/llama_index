from unittest.mock import AsyncMock

import pytest

from llama_index.core.evaluation.answer_relevancy import (
    AnswerRelevancyEvaluator,
    _default_parser_function,
)
from llama_index.core.llms.mock import MockLLM


@pytest.mark.parametrize("score", [0, 1, 2, 10, 12, 100])
def test_parser_preserves_integer_scores(score: int) -> None:
    assert _default_parser_function(f"Feedback [RESULT] {score}") == (
        float(score),
        "Feedback",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("score_threshold", [2, 10])
async def test_evaluator_normalizes_full_scale_scores(
    monkeypatch: pytest.MonkeyPatch, score_threshold: int
) -> None:
    raw_response = f"Feedback [RESULT] {score_threshold}"
    monkeypatch.setattr(MockLLM, "apredict", AsyncMock(return_value=raw_response))
    evaluator = AnswerRelevancyEvaluator(
        llm=MockLLM(),
        eval_template=f"Rate on a scale of 0 to {score_threshold}. Query: {{query}} Response: {{response}}",
        score_threshold=score_threshold,
    )

    result = await evaluator.aevaluate(query="query", response="answer")
    assert result.score == 1.0
    assert result.invalid_result is False
    assert result.feedback == raw_response
