import pytest

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.pairwise import (
    EvaluationSource,
    PairwiseComparisonEvaluator,
)
from llama_index.core.llms.mock import MockLLM


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flipped_score", "expected_passing", "expected_score"),
    [
        # flipped judge picked "response" (answer_2 in the flipped call),
        # so in the original frame "response" won.
        (0.0, True, 1.0),
        # flipped judge picked "second_response" (answer_1 in the flipped
        # call), so in the original frame "second_response" won.
        (1.0, False, 0.0),
    ],
)
async def test_resolve_results_converts_flipped_frame(
    flipped_score: float, expected_passing: bool, expected_score: float
) -> None:
    """
    When the original judge ties ([[C]]) and the flipped judge is decisive,
    _resolve_results must return the flipped judge's verdict converted into
    the original (response vs. second_response) frame, not the raw flipped
    verdict.
    """
    evaluator = PairwiseComparisonEvaluator(llm=MockLLM())

    eval_result = EvaluationResult(
        query="q", response="tie reasoning", passing=None, score=0.5
    )
    flipped_eval_result = EvaluationResult(
        query="q", response="flipped reasoning", passing=None, score=flipped_score
    )

    resolved = await evaluator._resolve_results(eval_result, flipped_eval_result)

    assert resolved.pairwise_source == EvaluationSource.FLIPPED
    assert resolved.passing is expected_passing
    assert resolved.score == expected_score
