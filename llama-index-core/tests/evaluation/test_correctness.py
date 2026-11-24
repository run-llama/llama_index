"""Tests for CorrectnessEvaluator reference requirements."""

from typing import Any

import pytest
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM


class RecordingJudge(CustomLLM):
    last_prompt: str = ""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="recording-judge")

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kw: Any
    ) -> CompletionResponse:
        self.last_prompt = prompt
        return CompletionResponse(text="5.0\nLooks right.")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kw: Any):
        yield self.complete(prompt)


def test_correctness_evaluator_requires_reference() -> None:
    judge = RecordingJudge()
    evaluator = CorrectnessEvaluator(llm=judge)

    with pytest.raises(ValueError, match="reference must be provided"):
        evaluator.evaluate(
            query="What is the capital of Australia?",
            response="Canberra",
            reference=None,
        )

    with pytest.raises(ValueError, match="reference must be provided"):
        evaluator.evaluate(
            query="What is the capital of Australia?",
            response="Canberra",
            reference="",
        )


def test_correctness_evaluator_with_reference() -> None:
    judge = RecordingJudge()
    evaluator = CorrectnessEvaluator(llm=judge)

    result = evaluator.evaluate(
        query="What is the capital of Australia?",
        response="Canberra",
        reference="Canberra",
    )

    assert result.score == 5.0
    assert result.passing is True
    assert "(NO REFERENCE ANSWER SUPPLIED)" not in judge.last_prompt
    assert "Canberra" in judge.last_prompt
