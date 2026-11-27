from unittest.mock import AsyncMock

import pytest

from llama_index.core.evaluation.pairwise import PairwiseComparisonEvaluator
from llama_index.core.llms.mock import MockLLM


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["[[A]]", "[[B]]", "[[C]]"])
async def test_invalid_flipped_judge_preserves_feedback(
    monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    original = f"Original reasoning {verdict}"
    flipped = "The judge could not produce a verdict."
    predict = AsyncMock(side_effect=[original, flipped])
    monkeypatch.setattr(MockLLM, "apredict", predict)

    result = await PairwiseComparisonEvaluator(llm=MockLLM()).aevaluate(
        query="query", response="answer", second_response="alternative"
    )

    assert predict.await_count == 2
    assert result.invalid_result is True
    assert result.feedback == flipped
    assert result.response == original
    assert result.score is None


@pytest.mark.asyncio
async def test_invalid_original_judge_preserves_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback = "The judge could not produce a verdict."
    predict = AsyncMock(return_value=feedback)
    monkeypatch.setattr(MockLLM, "apredict", predict)

    result = await PairwiseComparisonEvaluator(llm=MockLLM()).aevaluate(
        query="query", response="answer", second_response="alternative"
    )

    assert predict.await_count == 1
    assert result.invalid_result is True
    assert result.feedback == feedback


@pytest.mark.asyncio
async def test_valid_consensus_preserves_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict = AsyncMock(side_effect=["First reasoning [[A]]", "Second reasoning [[B]]"])
    monkeypatch.setattr(MockLLM, "apredict", predict)

    result = await PairwiseComparisonEvaluator(llm=MockLLM()).aevaluate(
        query="query", response="answer", second_response="alternative"
    )

    assert result.invalid_result is False
    assert result.passing is True
    assert result.score == 1.0
