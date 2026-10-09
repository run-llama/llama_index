"""Pairwise evaluation tests."""

from typing import Any, List, Optional, Tuple

import pytest
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.pairwise import (
    EvaluationSource,
    PairwiseComparisonEvaluator,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import BasePromptTemplate
from pydantic import Field


class MockPairwiseJudgeLLM(MockLLM):
    """
    Mock LLM that returns scripted judge verdicts in call order.

    Records the answer order shown on each call so tests can assert the
    flipped evaluation really saw the answers swapped.
    """

    verdicts: List[str] = Field(default_factory=list)
    shown_answer_orders: List[Tuple[str, str]] = Field(default_factory=list)

    async def apredict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        self.shown_answer_orders.append(
            (prompt_args["answer_1"], prompt_args["answer_2"])
        )
        return self.verdicts.pop(0)


def _evaluate(
    verdicts: List[str], enforce_consensus: bool = True
) -> Tuple[EvaluationResult, MockPairwiseJudgeLLM]:
    judge = MockPairwiseJudgeLLM()
    judge.verdicts = list(verdicts)
    evaluator = PairwiseComparisonEvaluator(
        llm=judge, enforce_consensus=enforce_consensus
    )
    result = evaluator.evaluate(
        query="query",
        response="response",
        second_response="second_response",
        reference="reference",
    )
    return result, judge


@pytest.mark.parametrize(
    (
        "original_verdict",
        "flipped_verdict",
        "expected_passing",
        "expected_score",
        "expected_source",
    ),
    [
        # both judges agree response wins
        ("[[A]]", "[[B]]", True, 1.0, EvaluationSource.ORIGINAL),
        # both judges agree second_response wins
        ("[[B]]", "[[A]]", False, 0.0, EvaluationSource.ORIGINAL),
        # original judge decisive, flipped judge ties
        ("[[A]]", "[[C]]", True, 1.0, EvaluationSource.ORIGINAL),
        ("[[B]]", "[[C]]", False, 0.0, EvaluationSource.ORIGINAL),
        # both judges tie
        ("[[C]]", "[[C]]", None, 0.5, EvaluationSource.ORIGINAL),
        # judges disagree: position bias, inconclusive
        ("[[A]]", "[[A]]", None, 0.5, EvaluationSource.NEITHER),
        ("[[B]]", "[[B]]", None, 0.5, EvaluationSource.NEITHER),
        # original judge ties, flipped judge decisive:
        # verdict must be converted back to the original frame
        ("[[C]]", "[[B]]", True, 1.0, EvaluationSource.FLIPPED),
        ("[[C]]", "[[A]]", False, 0.0, EvaluationSource.FLIPPED),
    ],
)
def test_resolve_results_returns_original_frame_verdict(
    original_verdict: str,
    flipped_verdict: str,
    expected_passing: Optional[bool],
    expected_score: float,
    expected_source: EvaluationSource,
) -> None:
    result, judge = _evaluate([original_verdict, flipped_verdict])

    assert result.passing == expected_passing
    assert result.score == expected_score
    assert result.pairwise_source == expected_source
    # score/passing are always expressed in the original frame:
    # 1.0/True means `response` beat `second_response`
    if expected_source == EvaluationSource.FLIPPED:
        assert result.feedback == flipped_verdict
    elif expected_source == EvaluationSource.ORIGINAL:
        assert result.feedback == original_verdict
    # the flipped evaluation must see the answers swapped
    assert judge.shown_answer_orders == [
        ("response", "second_response"),
        ("second_response", "response"),
    ]


@pytest.mark.parametrize(
    ("verdict", "expected_passing", "expected_score"),
    [
        ("[[A]]", True, 1.0),
        ("[[B]]", False, 0.0),
        ("[[C]]", None, 0.5),
    ],
)
def test_no_consensus_returns_single_judgement(
    verdict: str, expected_passing: Optional[bool], expected_score: float
) -> None:
    result, judge = _evaluate([verdict], enforce_consensus=False)

    assert result.passing == expected_passing
    assert result.score == expected_score
    assert result.pairwise_source == EvaluationSource.ORIGINAL
    assert judge.shown_answer_orders == [("response", "second_response")]
