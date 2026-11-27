from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.evaluation.context_relevancy import (
    ContextRelevancyEvaluator,
    _default_parser_function,
)
from llama_index.core.llms.mock import MockLLM


@pytest.mark.parametrize("score", [".", "..", "1.2.3"])
def test_parser_rejects_malformed_scores(score: str) -> None:
    assert _default_parser_function(f"Feedback [RESULT] {score}") == (None, None)


@pytest.mark.parametrize("score", ["0", "2.5", "4.0"])
def test_parser_preserves_valid_scores(score: str) -> None:
    assert _default_parser_function(f"Feedback [RESULT] {score}") == (
        float(score),
        "Feedback",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_error", [False, True])
@pytest.mark.parametrize("score", [".", "1.2.3"])
async def test_malformed_scores_follow_invalid_result_policy(
    monkeypatch: pytest.MonkeyPatch, score: str, raise_error: bool
) -> None:
    raw_response = f"Feedback [RESULT] {score}"
    query_engine = MagicMock()
    query_engine.aquery = AsyncMock(return_value=raw_response)
    index = MagicMock()
    index.as_query_engine.return_value = query_engine
    monkeypatch.setattr(
        "llama_index.core.evaluation.context_relevancy.SummaryIndex.from_documents",
        MagicMock(return_value=index),
    )
    evaluator = ContextRelevancyEvaluator(llm=MockLLM(), raise_error=raise_error)

    if raise_error:
        with pytest.raises(ValueError, match="^The response is invalid$"):
            await evaluator.aevaluate(query="query", contexts=["context"])
    else:
        result = await evaluator.aevaluate(query="query", contexts=["context"])
        assert result.invalid_result is True
        assert result.invalid_reason == "Unable to parse the output string."
        assert result.score is None
        assert result.feedback == raw_response
