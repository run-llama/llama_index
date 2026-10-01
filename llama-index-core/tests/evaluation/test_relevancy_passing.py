"""`passing` must reflect the score for the relevancy evaluators.

Both evaluators normalise `score` by `score_threshold`, so a normalised score
of 1.0 means the raw score reached the threshold. `passing` has to say so:
`EvaluationResult.passing` defaults to None, and callers that gate on it, such
as `EvalQueryEngineTool._process_tool_output`, treat None as a failure. Without
these assertions a maximum-score response and a zero-score one are
indistinguishable downstream.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.evaluation.answer_relevancy import AnswerRelevancyEvaluator
from llama_index.core.evaluation.context_relevancy import ContextRelevancyEvaluator

# The default prompts score two criteria at one point each, and the default
# threshold is 2.0, so "[RESULT] 2" is the maximum and "[RESULT] 0" the minimum.
FULL_MARKS = "Matches the subject and addresses the focus.\n[RESULT] 2"
HALF_MARKS = "Matches the subject but not the focus.\n[RESULT] 1"
NO_MARKS = "Off topic entirely.\n[RESULT] 0"
UNPARSEABLE = "I am unable to score this."


def _answer_evaluator(reply: str) -> AnswerRelevancyEvaluator:
    return AnswerRelevancyEvaluator(
        llm=AsyncMock(**{"apredict.return_value": reply})
    )


def _context_evaluator(
    monkeypatch: pytest.MonkeyPatch, reply: str
) -> ContextRelevancyEvaluator:
    """Stub the index/query-engine pipeline so only the scoring logic is under test."""
    query_engine = MagicMock()
    query_engine.aquery = AsyncMock(return_value=reply)

    index = MagicMock()
    index.as_query_engine.return_value = query_engine

    summary_index = MagicMock()
    summary_index.from_documents.return_value = index
    monkeypatch.setattr(
        "llama_index.core.evaluation.context_relevancy.SummaryIndex", summary_index
    )

    return ContextRelevancyEvaluator(llm=AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected_score", "expected_passing"),
    [
        (FULL_MARKS, 1.0, True),
        (HALF_MARKS, 0.5, False),
        (NO_MARKS, 0.0, False),
    ],
)
async def test_answer_relevancy_sets_passing(
    reply: str, expected_score: float, expected_passing: bool
) -> None:
    result = await _answer_evaluator(reply).aevaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
    )
    assert result.score == expected_score
    assert result.passing is expected_passing


@pytest.mark.asyncio
async def test_answer_relevancy_passing_is_none_when_unparseable() -> None:
    """An unscored result must not claim to have passed or failed."""
    result = await _answer_evaluator(UNPARSEABLE).aevaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
    )
    assert result.invalid_result is True
    assert result.passing is None


# The context prompt scores two criteria at two points each, threshold 4.0.
CONTEXT_FULL_MARKS = "The context fully answers the query.\n[RESULT] 4"
CONTEXT_PARTIAL_MARKS = "The context is only partly relevant.\n[RESULT] 2"
CONTEXT_NO_MARKS = "The context is unrelated.\n[RESULT] 0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected_score", "expected_passing"),
    [
        (CONTEXT_FULL_MARKS, 1.0, True),
        (CONTEXT_PARTIAL_MARKS, 0.5, False),
        (CONTEXT_NO_MARKS, 0.0, False),
    ],
)
async def test_context_relevancy_sets_passing(
    monkeypatch: pytest.MonkeyPatch,
    reply: str,
    expected_score: float,
    expected_passing: bool,
) -> None:
    result = await _context_evaluator(monkeypatch, reply).aevaluate(
        query="What is the capital of France?",
        contexts=["Paris is the capital of France."],
    )
    assert result.score == expected_score
    assert result.passing is expected_passing
