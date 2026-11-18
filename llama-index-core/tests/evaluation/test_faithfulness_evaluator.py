"""Tests for FaithfulnessEvaluator input handling (issue #24275)."""

from typing import Any, Dict, Iterator, List

import pytest

import llama_index.core as core
from llama_index.core import Settings
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.node_parser import TokenTextSplitter
from tests.mock_utils.mock_utils import mock_tokenizer

QUERY = "What was the revenue?"
RESPONSE = "The revenue was 100."
CONTEXTS = ["The revenue was 100."]


@pytest.fixture()
def _tokenizer() -> Iterator[None]:
    # Let SummaryIndex split documents without a real tokenizer (tiktoken).
    old_tokenizer = core.global_tokenizer
    old_transformations = Settings._transformations
    core.global_tokenizer = mock_tokenizer
    Settings._transformations = [TokenTextSplitter(tokenizer=mock_tokenizer)]
    yield
    core.global_tokenizer = old_tokenizer
    Settings._transformations = old_transformations


def _make_stub_llm(reply: str, counter: Dict[str, int]) -> CustomLLM:
    class StubLLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(model_name="stub")

        @llm_completion_callback()
        def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            counter["calls"] += 1
            return CompletionResponse(text=reply)

        @llm_completion_callback()
        def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> Any:
            raise NotImplementedError

    return StubLLM()


def _evaluator(counter: Dict[str, int]) -> FaithfulnessEvaluator:
    return FaithfulnessEvaluator(llm=_make_stub_llm("YES", counter))


@pytest.mark.parametrize("contexts", [[], [""], ["", ""]])
def test_empty_contexts_are_invalid_not_scored(contexts: List[str]) -> None:
    counter = {"calls": 0}
    result = _evaluator(counter).evaluate(
        query=QUERY, response=RESPONSE, contexts=contexts
    )
    assert counter["calls"] == 0
    assert result.invalid_result is True
    assert result.invalid_reason
    assert result.passing is None
    assert result.score is None


def test_empty_response_is_invalid_not_scored() -> None:
    counter = {"calls": 0}
    result = _evaluator(counter).evaluate(query=QUERY, response="", contexts=CONTEXTS)
    assert counter["calls"] == 0
    assert result.invalid_result is True
    assert result.invalid_reason
    assert result.passing is None
    assert result.score is None


def test_none_inputs_raise_value_error() -> None:
    evaluator = _evaluator({"calls": 0})
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.evaluate(query=QUERY, response=None, contexts=CONTEXTS)
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.evaluate(query=QUERY, response=RESPONSE, contexts=None)


@pytest.mark.usefixtures("_tokenizer")
def test_mixed_empty_and_nonempty_contexts_are_judged() -> None:
    counter = {"calls": 0}
    result = _evaluator(counter).evaluate(
        query=QUERY, response=RESPONSE, contexts=["", *CONTEXTS]
    )
    assert counter["calls"] >= 1
    assert result.invalid_result is False
    assert result.passing is True
    assert result.score == 1.0


@pytest.mark.usefixtures("_tokenizer")
def test_valid_inputs_are_judged() -> None:
    counter = {"calls": 0}
    result = _evaluator(counter).evaluate(
        query=QUERY, response=RESPONSE, contexts=CONTEXTS
    )
    assert counter["calls"] >= 1
    assert result.invalid_result is False
    assert result.passing is True
    assert result.score == 1.0
