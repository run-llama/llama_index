"""Tests for env/runtime/tools.py: evaluate_rubric, score_examples wrappers."""

import json

import pytest

from environments.rubric_discovery.env.evaluation.service import EvaluationService
from environments.rubric_discovery.env.runtime.tools import (
    ToolContext,
    score_examples,
    evaluate_rubric,
)
from environments.rubric_discovery.env.types import LabeledExample, RubricDiscoveryConfig


@pytest.fixture
def tool_ctx() -> ToolContext:
    config = RubricDiscoveryConfig(eval_backend="subprocess")
    examples = [
        LabeledExample("What is 2+2?", "4", 1.0),
        LabeledExample("What is 2+2?", "5", 0.0),
        LabeledExample("What is 3+3?", "6", 1.0),
    ]
    service = EvaluationService(config)
    return ToolContext(config=config, train_examples=examples, eval_service=service)


class TestEvaluateRubric:
    def test_valid_rubric(self, tool_ctx: ToolContext) -> None:
        code = 'def rubric_fn(input_text, response): return 0.5'
        result = json.loads(evaluate_rubric(tool_ctx, code))
        assert result["success"]
        assert result["num_examples"] == 3
        assert "accuracy" in result
        assert "mae" in result
        assert len(result["details"]) == 3

    def test_no_rubric_fn(self, tool_ctx: ToolContext) -> None:
        code = "print('hello')"
        result = json.loads(evaluate_rubric(tool_ctx, code))
        assert not result["success"]
        assert "error" in result

    def test_invalid_signature(self, tool_ctx: ToolContext) -> None:
        code = "def rubric_fn(x): return 0.5"
        result = json.loads(evaluate_rubric(tool_ctx, code))
        assert not result["success"]

    def test_increments_tool_count(self, tool_ctx: ToolContext) -> None:
        assert tool_ctx.tool_call_count == 0
        code = "def rubric_fn(input_text, response): return 0.5"
        evaluate_rubric(tool_ctx, code)
        assert tool_ctx.tool_call_count == 1

    def test_updates_latest_source(self, tool_ctx: ToolContext) -> None:
        code = "def rubric_fn(input_text, response): return 0.5"
        evaluate_rubric(tool_ctx, code)
        assert tool_ctx.latest_source is not None
        assert "rubric_fn" in tool_ctx.latest_source


class TestScoreExamples:
    def test_all_examples(self, tool_ctx: ToolContext) -> None:
        code = "def rubric_fn(input_text, response): return 0.5"
        result = json.loads(score_examples(tool_ctx, code))
        assert result["success"]
        assert len(result["scores"]) == 3

    def test_subset(self, tool_ctx: ToolContext) -> None:
        code = "def rubric_fn(input_text, response): return 0.5"
        result = json.loads(score_examples(tool_ctx, code, indices=[0, 2]))
        assert result["success"]
        assert len(result["scores"]) == 2

    def test_no_rubric_fn(self, tool_ctx: ToolContext) -> None:
        result = json.loads(score_examples(tool_ctx, "x = 1"))
        assert not result["success"]

    def test_empty_indices(self, tool_ctx: ToolContext) -> None:
        code = "def rubric_fn(input_text, response): return 0.5"
        result = json.loads(score_examples(tool_ctx, code, indices=[99]))
        assert not result["success"]
