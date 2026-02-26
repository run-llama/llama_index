"""Tests for env/types.py: shared type contracts."""

import pytest

from environments.rubric_discovery.env.types import (
    DatasetRow,
    EvaluationResult,
    ExecutionBackend,
    LabeledExample,
    RubricDiscoveryConfig,
)


class TestLabeledExample:
    def test_create_minimal(self) -> None:
        ex = LabeledExample(input_text="hello", response="world", score=0.5)
        assert ex.input_text == "hello"
        assert ex.response == "world"
        assert ex.score == 0.5
        assert ex.category == ""
        assert ex.metadata == {}

    def test_create_full(self) -> None:
        ex = LabeledExample(
            input_text="q", response="a", score=1.0,
            category="test", metadata={"key": "val"},
        )
        assert ex.category == "test"
        assert ex.metadata == {"key": "val"}

    def test_frozen(self) -> None:
        ex = LabeledExample(input_text="q", response="a", score=0.0)
        with pytest.raises(AttributeError):
            ex.score = 0.5  # type: ignore[misc]

    def test_to_dict_minimal(self) -> None:
        ex = LabeledExample(input_text="q", response="a", score=0.5)
        d = ex.to_dict()
        assert d == {"input_text": "q", "response": "a", "score": 0.5}

    def test_to_dict_full(self) -> None:
        ex = LabeledExample(
            input_text="q", response="a", score=1.0,
            category="cat", metadata={"k": "v"},
        )
        d = ex.to_dict()
        assert d["category"] == "cat"
        assert d["metadata"] == {"k": "v"}

    def test_roundtrip(self) -> None:
        ex = LabeledExample(
            input_text="q", response="a", score=0.75,
            category="c", metadata={"x": 1},
        )
        d = ex.to_dict()
        ex2 = LabeledExample.from_dict(d)
        assert ex2 == ex

    def test_from_dict_missing_optional(self) -> None:
        d = {"input_text": "q", "response": "a", "score": "0.3"}
        ex = LabeledExample.from_dict(d)
        assert ex.score == 0.3
        assert ex.category == ""


class TestDatasetRow:
    def test_roundtrip(self) -> None:
        train = [LabeledExample("q1", "a1", 0.5)]
        test = [LabeledExample("q2", "a2", 1.0)]
        row = DatasetRow(
            train_examples=train, test_examples=test,
            category="cat", metadata={"key": "val"},
        )
        d = row.to_dict()
        row2 = DatasetRow.from_dict(d)
        assert len(row2.train_examples) == 1
        assert len(row2.test_examples) == 1
        assert row2.category == "cat"


class TestEvaluationResult:
    def test_defaults(self) -> None:
        r = EvaluationResult()
        assert r.predictions == []
        assert r.labels == []
        assert r.valid is True
        assert r.error is None

    def test_invalid(self) -> None:
        r = EvaluationResult(valid=False, error="bad")
        assert not r.valid
        assert r.error == "bad"


class TestExecutionBackend:
    def test_values(self) -> None:
        assert ExecutionBackend.AUTO == "auto"
        assert ExecutionBackend.SANDBOX == "sandbox"
        assert ExecutionBackend.SUBPROCESS == "subprocess"


class TestRubricDiscoveryConfig:
    def test_defaults(self) -> None:
        cfg = RubricDiscoveryConfig()
        assert cfg.rlm_model == "gpt-4.1-mini"
        assert cfg.max_turns == 10
        assert cfg.eval_tolerance == 0.3
        assert cfg.eval_backend == "auto"

    def test_from_dict(self) -> None:
        cfg = RubricDiscoveryConfig.from_dict({
            "max_turns": 5,
            "eval_backend": "subprocess",
            "unknown_key": "ignored",
        })
        assert cfg.max_turns == 5
        assert cfg.eval_backend == "subprocess"

    def test_get_execution_backend(self) -> None:
        cfg = RubricDiscoveryConfig(eval_backend="subprocess")
        assert cfg.get_execution_backend() == ExecutionBackend.SUBPROCESS
