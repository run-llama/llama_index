from typing import Any

import pytest

from llama_index.core.evaluation.correctness import CorrectnessEvaluator
from llama_index.core.llms.mock import MockLLM


class StubLLM(MockLLM):
    """Mock LLM whose judge reply is fixed, bypassing prompt plumbing."""

    _response: str = ""

    def __init__(self, response: str) -> None:
        super().__init__()
        self._response = response

    async def apredict(self, prompt: Any, **kwargs: Any) -> str:
        return self._response


def _evaluate(judge_reply: str):
    evaluator = CorrectnessEvaluator(llm=StubLLM(judge_reply))
    return evaluator.evaluate(query="test query", response="test response")


@pytest.mark.parametrize("judge_reply", ["10\nok", "inf\nok", "nan\nok", "-3\nok"])
def test_out_of_contract_scores_are_invalid(judge_reply: str) -> None:
    result = _evaluate(judge_reply)
    assert result.invalid_result is True
    assert result.invalid_reason
    assert result.passing is False


def test_in_range_score_is_unchanged() -> None:
    result = _evaluate("4.0\nok")
    assert result.invalid_result is False
    assert result.invalid_reason is None
    assert result.score == 4.0
    assert result.passing is True


def test_custom_parser_scores_are_not_validated() -> None:
    evaluator = CorrectnessEvaluator(
        llm=StubLLM("ignored"),
        parser_function=lambda _: (10.0, "custom scale"),
    )
    result = evaluator.evaluate(query="test query", response="test response")
    assert result.invalid_result is False
    assert result.score == 10.0
