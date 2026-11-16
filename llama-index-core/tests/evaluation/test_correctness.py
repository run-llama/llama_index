"""Tests for CorrectnessEvaluator, pinning down its reference handling (issue #24223)."""

from typing import Any

import pytest

from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

PLACEHOLDER = "(NO REFERENCE ANSWER SUPPLIED)"


class RecordingJudge(CustomLLM):
    """LLM double that records the prompt it was given and always scores 5."""

    last_prompt: str = ""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="recording-judge")

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.last_prompt = prompt
        return CompletionResponse(text="5.0\nLooks right.")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        yield self.complete(prompt, formatted=formatted, **kwargs)


@pytest.mark.parametrize("reference", [None, ""])
def test_missing_reference_is_scored_reference_free(reference: Any) -> None:
    """Omitting the reference does not raise; the judge sees a placeholder."""
    judge = RecordingJudge()
    result = CorrectnessEvaluator(llm=judge).evaluate(
        query="What is the capital of Australia?",
        response="Canberra",
        reference=reference,
    )

    assert result.score == 5.0
    assert result.passing
    assert PLACEHOLDER in judge.last_prompt


def test_reference_is_forwarded_to_the_judge() -> None:
    """A provided reference reaches the judge and replaces the placeholder."""
    judge = RecordingJudge()
    result = CorrectnessEvaluator(llm=judge).evaluate(
        query="What is the capital of Australia?",
        response="Canberra",
        reference="Canberra",
    )

    assert result.score == 5.0
    assert PLACEHOLDER not in judge.last_prompt
    assert "Canberra" in judge.last_prompt


def test_query_and_response_are_still_required() -> None:
    """Unlike the reference, a missing query or response keeps raising."""
    evaluator = CorrectnessEvaluator(llm=RecordingJudge())

    with pytest.raises(ValueError, match="query, and response must be provided"):
        evaluator.evaluate(query=None, response="Canberra", reference="Canberra")

    with pytest.raises(ValueError, match="query, and response must be provided"):
        evaluator.evaluate(query="What is the capital of Australia?", response=None)
