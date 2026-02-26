"""Tests for env/evaluation/: backends, service, runner."""

import json

import pytest

from environments.rubric_discovery.env.evaluation.backends import (
    SubprocessBackend,
    _build_harness,
    resolve_backend,
)
from environments.rubric_discovery.env.evaluation.service import EvaluationService
from environments.rubric_discovery.env.types import (
    ExecutionBackend,
    LabeledExample,
    RubricDiscoveryConfig,
)


class TestBuildHarness:
    def test_harness_structure(self) -> None:
        examples = [
            LabeledExample("q1", "a1", 0.5),
            LabeledExample("q2", "a2", 1.0),
        ]
        source = "def rubric_fn(input_text, response): return 0.5"
        harness = _build_harness(source, examples)
        assert "def rubric_fn" in harness
        assert "json.dumps" in harness
        assert "q1" in harness


class TestSubprocessBackend:
    def test_simple_execution(self) -> None:
        backend = SubprocessBackend()
        examples = [
            LabeledExample("q1", "a1", 0.5),
            LabeledExample("q2", "a2", 1.0),
        ]
        source = "def rubric_fn(input_text, response): return 0.5"
        result = backend.execute(source, examples, timeout_s=10)
        assert result.valid
        assert len(result.predictions) == 2
        assert all(p == 0.5 for p in result.predictions)
        assert result.labels == [0.5, 1.0]

    def test_runtime_error_clamped(self) -> None:
        """Rubric functions that error should produce 0.0 predictions."""
        backend = SubprocessBackend()
        examples = [LabeledExample("q", "a", 0.5)]
        source = "def rubric_fn(input_text, response): return 1/0"
        result = backend.execute(source, examples, timeout_s=10)
        assert result.valid
        assert result.predictions == [0.0]

    def test_timeout(self) -> None:
        backend = SubprocessBackend()
        examples = [LabeledExample("q", "a", 0.5)]
        source = "import time\ndef rubric_fn(input_text, response):\n    time.sleep(30)\n    return 0.5"
        result = backend.execute(source, examples, timeout_s=2)
        assert not result.valid
        assert "timed out" in result.error

    def test_score_clamping(self) -> None:
        """Scores outside [0,1] should be clamped."""
        backend = SubprocessBackend()
        examples = [LabeledExample("q", "a", 0.5)]
        source = "def rubric_fn(input_text, response): return 5.0"
        result = backend.execute(source, examples, timeout_s=10)
        assert result.valid
        assert result.predictions == [1.0]  # clamped


class TestResolveBackend:
    def test_subprocess(self) -> None:
        backend = resolve_backend(ExecutionBackend.SUBPROCESS)
        assert isinstance(backend, SubprocessBackend)

    def test_auto_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRIME_API_KEY", raising=False)
        backend = resolve_backend(ExecutionBackend.AUTO)
        assert isinstance(backend, SubprocessBackend)


class TestEvaluationService:
    def setup_method(self) -> None:
        self.config = RubricDiscoveryConfig(eval_backend="subprocess")
        self.service = EvaluationService(self.config)

    def test_extract_source(self) -> None:
        text = "```python\ndef rubric_fn(input_text, response): return 0.5\n```"
        source = self.service.extract_source(text)
        assert source is not None
        assert "rubric_fn" in source

    def test_validate_valid(self) -> None:
        source = "def rubric_fn(input_text, response): return 0.5"
        result = self.service.validate(source)
        assert result["valid"]
        assert result["signature_ok"]
        assert result["callable_ok"]

    def test_validate_bad_signature(self) -> None:
        source = "def rubric_fn(x): return 0.5"
        result = self.service.validate(source)
        assert not result["valid"]
        assert not result["signature_ok"]

    def test_evaluate(self) -> None:
        examples = [
            LabeledExample("q1", "a1", 0.5),
            LabeledExample("q2", "a2", 1.0),
        ]
        source = "def rubric_fn(input_text, response): return 0.5"
        result = self.service.evaluate(source, examples)
        assert result.valid
        assert len(result.predictions) == 2

    def test_score_within_tolerance(self) -> None:
        from environments.rubric_discovery.env.types import EvaluationResult

        result = EvaluationResult(
            predictions=[0.5, 0.5], labels=[0.5, 1.0], valid=True
        )
        score = self.service.score_within_tolerance(result)
        # First is exact match (within 0.3), second is 0.5 away (not within 0.3)
        assert score == 0.5
