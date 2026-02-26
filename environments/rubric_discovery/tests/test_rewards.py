"""Tests for env/runtime/rewards.py: reward functions."""

import math

import pytest

from environments.rubric_discovery.env.runtime.rewards import (
    RewardBreakdown,
    calibration_reward,
    compute_rewards,
    discrimination_reward,
    generalization_reward,
    iteration_reward,
    validity_reward,
)
from environments.rubric_discovery.env.types import EvaluationResult


class TestGeneralizationReward:
    def test_perfect(self) -> None:
        result = EvaluationResult(
            predictions=[0.5, 1.0, 0.0],
            labels=[0.5, 1.0, 0.0],
            valid=True,
        )
        assert generalization_reward(result, tolerance=0.3) == 1.0

    def test_all_wrong(self) -> None:
        result = EvaluationResult(
            predictions=[0.0, 0.0, 0.0],
            labels=[1.0, 1.0, 1.0],
            valid=True,
        )
        assert generalization_reward(result, tolerance=0.3) == 0.0

    def test_partial(self) -> None:
        result = EvaluationResult(
            predictions=[0.5, 0.5, 0.5],
            labels=[0.5, 1.0, 0.0],
            valid=True,
        )
        reward = generalization_reward(result, tolerance=0.3)
        # Only first is within tolerance
        assert reward == pytest.approx(1 / 3, abs=0.01)

    def test_invalid(self) -> None:
        result = EvaluationResult(valid=False)
        assert generalization_reward(result) == 0.0


class TestCalibrationReward:
    def test_perfect(self) -> None:
        result = EvaluationResult(
            predictions=[0.5, 1.0], labels=[0.5, 1.0], valid=True
        )
        assert calibration_reward(result) == 1.0

    def test_worst(self) -> None:
        result = EvaluationResult(
            predictions=[0.0, 0.0], labels=[1.0, 1.0], valid=True
        )
        assert calibration_reward(result) == 0.0

    def test_invalid(self) -> None:
        assert calibration_reward(EvaluationResult(valid=False)) == 0.0


class TestDiscriminationReward:
    def test_constant(self) -> None:
        result = EvaluationResult(
            predictions=[0.5, 0.5, 0.5], labels=[0, 0.5, 1], valid=True
        )
        assert discrimination_reward(result) == 0.0

    def test_varied(self) -> None:
        result = EvaluationResult(
            predictions=[0.0, 0.5, 1.0], labels=[0, 0.5, 1], valid=True
        )
        reward = discrimination_reward(result)
        assert reward > 0.5

    def test_single(self) -> None:
        result = EvaluationResult(
            predictions=[0.5], labels=[0.5], valid=True
        )
        assert discrimination_reward(result) == 0.0

    def test_invalid(self) -> None:
        assert discrimination_reward(EvaluationResult(valid=False)) == 0.0


class TestValidityReward:
    def test_none(self) -> None:
        assert validity_reward(None) == 0.0

    def test_no_rubric(self) -> None:
        assert validity_reward("def other(): pass") == 0.0

    def test_bad_signature(self) -> None:
        assert validity_reward("def rubric_fn(x): return 0.5") == 0.25

    def test_runtime_error(self) -> None:
        source = "def rubric_fn(input_text, response): return 1/0"
        assert validity_reward(source) == 0.5

    def test_fully_valid(self) -> None:
        source = "def rubric_fn(input_text, response): return 0.5"
        assert validity_reward(source) == 1.0


class TestIterationReward:
    def test_zero_calls(self) -> None:
        assert iteration_reward(0) == 0.0

    def test_one_call(self) -> None:
        reward = iteration_reward(1)
        assert 0.2 < reward < 0.5

    def test_many_calls(self) -> None:
        reward = iteration_reward(10)
        assert reward > 0.8

    def test_monotonic(self) -> None:
        rewards = [iteration_reward(i) for i in range(10)]
        for i in range(1, len(rewards)):
            assert rewards[i] >= rewards[i - 1]


class TestComputeRewards:
    def test_perfect_run(self) -> None:
        result = EvaluationResult(
            predictions=[0.5, 1.0], labels=[0.5, 1.0], valid=True
        )
        source = "def rubric_fn(input_text, response): return 0.5"
        breakdown = compute_rewards(result, source, num_tool_calls=5)
        assert breakdown.total > 0.5
        assert breakdown.generalization == 1.0
        assert breakdown.validity == 1.0

    def test_invalid_source(self) -> None:
        result = EvaluationResult(valid=False)
        breakdown = compute_rewards(result, None, num_tool_calls=0)
        assert breakdown.total == 0.0
        assert breakdown.generalization == 0.0
        assert breakdown.validity == 0.0

    def test_to_dict(self) -> None:
        bd = RewardBreakdown(
            generalization=0.8, calibration=0.7, discrimination=0.6,
            validity=1.0, iteration=0.5, total=0.75,
        )
        d = bd.to_dict()
        assert d["generalization"] == 0.8
        assert "total" in d
