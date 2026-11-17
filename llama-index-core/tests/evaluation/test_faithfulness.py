import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from llama_index.core import Settings
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import TokenTextSplitter

from tests.mock_utils.mock_utils import mock_tokenizer


@pytest.fixture()
def judge_llm(monkeypatch: pytest.MonkeyPatch, judge_response: str) -> MockLLM:
    monkeypatch.setattr("llama_index.core.global_tokenizer", mock_tokenizer)
    monkeypatch.setattr(
        Settings, "_transformations", [TokenTextSplitter(tokenizer=mock_tokenizer)]
    )
    monkeypatch.setattr(Settings, "_prompt_helper", None)
    monkeypatch.setattr(Settings, "_chat_prompt_helper", None)
    monkeypatch.setattr(Settings, "_callback_manager", None)
    monkeypatch.setattr(MockLLM, "predict", Mock(return_value=judge_response))
    monkeypatch.setattr(MockLLM, "apredict", AsyncMock(return_value=judge_response))
    return MockLLM()


def _evaluate(evaluator: FaithfulnessEvaluator, use_async: bool) -> EvaluationResult:
    kwargs: dict[str, Any] = {
        "query": "What was the revenue?",
        "response": "The revenue was 100.",
        "contexts": ["The revenue was 10."],
    }
    if use_async:
        return asyncio.run(evaluator.aevaluate(**kwargs))
    return evaluator.evaluate(**kwargs)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("judge_response", "passing"),
    [
        pytest.param("YES", True, id="yes"),
        pytest.param("yes.", True, id="period"),
        pytest.param("YES!", True, id="exclamation"),
        pytest.param(" \nYeS\t", True, id="case-and-whitespace"),
        pytest.param("NO", False, id="no"),
        pytest.param("yesterday", False, id="substring"),
        pytest.param("eyes", False, id="embedded-substring"),
        pytest.param("not yes", False, id="negated"),
        pytest.param(
            "NO. The source never says yes.", False, id="negative-explanation"
        ),
        pytest.param("YES or NO", False, id="ambiguous"),
        pytest.param("YES/NO", False, id="ambiguous-slash"),
        pytest.param("YES, the context supports it.", False, id="explanation"),
        pytest.param("Answer: YES", False, id="answer-prefix"),
        pytest.param("**YES**", False, id="markdown"),
        pytest.param("YES?", False, id="question"),
        pytest.param("YES..", False, id="repeated-punctuation"),
        pytest.param("YES !", False, id="internal-whitespace"),
        pytest.param("maybe", False, id="unknown"),
        pytest.param(" \t\n", False, id="whitespace"),
        pytest.param("", False, id="empty"),
    ],
)
def test_faithfulness_verdict(
    judge_llm: MockLLM, judge_response: str, passing: bool, use_async: bool
) -> None:
    result = _evaluate(FaithfulnessEvaluator(llm=judge_llm), use_async)

    # The real response synthesizer replaces an empty LLM answer with this text.
    feedback = judge_response or "Empty Response"
    assert result == EvaluationResult(
        query="What was the revenue?",
        response="The revenue was 100.",
        contexts=["The revenue was 10."],
        passing=passing,
        score=1.0 if passing else 0.0,
        feedback=feedback,
    )


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("judge_response", ["YES", "NO", "not yes", "maybe"])
def test_faithfulness_raise_error(
    judge_llm: MockLLM, judge_response: str, use_async: bool
) -> None:
    evaluator = FaithfulnessEvaluator(llm=judge_llm, raise_error=True)

    if judge_response == "YES":
        assert _evaluate(evaluator, use_async).passing is True
    else:
        with pytest.raises(ValueError, match="The response is invalid"):
            _evaluate(evaluator, use_async)
