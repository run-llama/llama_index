"""Reward functions for the rubric-discovery environment.

Reward weights (from spec):
    generalization_reward  (1.0): held-out within-tolerance rate
    calibration_reward     (0.4): 1 - MAE
    discrimination_reward  (0.3): penalizes constant scorers via prediction stdev
    validity_reward        (0.2): graded syntax/function/callability validity
    iteration_reward       (0.2): encourages iterative tool-assisted refinement
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from environments.rubric_discovery.env.candidate.extractor import (
    probe_callability,
    validate_signature,
)
from environments.rubric_discovery.env.types import EvaluationResult


@dataclass
class RewardBreakdown:
    """Detailed breakdown of all reward components.

    The final ``total`` is a weighted sum of individual rewards.
    """

    generalization: float = 0.0
    calibration: float = 0.0
    discrimination: float = 0.0
    validity: float = 0.0
    iteration: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return a dictionary of all reward components."""
        return {
            "generalization": self.generalization,
            "calibration": self.calibration,
            "discrimination": self.discrimination,
            "validity": self.validity,
            "iteration": self.iteration,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Weight constants
# ---------------------------------------------------------------------------
WEIGHT_GENERALIZATION = 1.0
WEIGHT_CALIBRATION = 0.4
WEIGHT_DISCRIMINATION = 0.3
WEIGHT_VALIDITY = 0.2
WEIGHT_ITERATION = 0.2


# ---------------------------------------------------------------------------
# Individual reward functions
# ---------------------------------------------------------------------------


def generalization_reward(
    result: EvaluationResult,
    tolerance: float = 0.3,
) -> float:
    """Held-out within-tolerance rate on hidden test examples.

    A prediction counts as correct if ``|pred - label| < tolerance``.
    """
    if not result.valid or not result.predictions:
        return 0.0
    correct = sum(
        1
        for p, l in zip(result.predictions, result.labels)
        if abs(p - l) < tolerance
    )
    return correct / len(result.labels)


def calibration_reward(result: EvaluationResult) -> float:
    """Score calibration: ``1 - MAE``.

    Rewards predictions that are close to the true scores,
    independent of the tolerance threshold.
    """
    if not result.valid or not result.predictions:
        return 0.0
    mae = sum(
        abs(p - l) for p, l in zip(result.predictions, result.labels)
    ) / len(result.labels)
    return max(0.0, 1.0 - mae)


def discrimination_reward(result: EvaluationResult) -> float:
    """Penalizes constant scorers via prediction standard deviation.

    A model that always outputs the same score gets 0.
    The reward is the stdev of predictions, capped at 1.0.
    """
    if not result.valid or not result.predictions or len(result.predictions) < 2:
        return 0.0
    mean = sum(result.predictions) / len(result.predictions)
    variance = sum((p - mean) ** 2 for p in result.predictions) / len(
        result.predictions
    )
    stdev = math.sqrt(variance)
    # Normalize: stdev of Uniform[0,1] is ~0.289, max for binary is 0.5
    # Cap at 1.0 after scaling by 2x
    return min(1.0, stdev * 2.0)


def validity_reward(source: Optional[str]) -> float:
    """Graded syntax/function/callability validity.

    Returns:
        1.0 — fully callable with correct signature
        0.5 — correct signature but fails at runtime
        0.25 — contains ``def rubric_fn`` but bad signature
        0.0 — no rubric_fn found at all
    """
    if source is None:
        return 0.0

    if "def rubric_fn" not in source:
        return 0.0

    sig_ok, _ = validate_signature(source)
    if not sig_ok:
        return 0.25

    call_ok, _ = probe_callability(source)
    if not call_ok:
        return 0.5

    return 1.0


def iteration_reward(num_tool_calls: int) -> float:
    """Encourages iterative tool-assisted refinement.

    Rewards increase with the number of tool calls, encouraging
    the model to test and refine its rubric rather than submitting
    a single attempt.

    Returns a value in [0.0, 1.0].
    """
    if num_tool_calls <= 0:
        return 0.0
    # Logarithmic scaling: 1 call → ~0.3, 3 calls → ~0.6, 7+ calls → ~1.0
    return min(1.0, math.log(1 + num_tool_calls) / math.log(8))


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------


def compute_rewards(
    result: EvaluationResult,
    source: Optional[str],
    num_tool_calls: int = 0,
    tolerance: float = 0.3,
) -> RewardBreakdown:
    """Compute all reward components and the weighted total.

    Args:
        result: Evaluation result from running rubric_fn on test examples.
        source: The rubric_fn source code (or None).
        num_tool_calls: Number of tool calls the model made.
        tolerance: Tolerance for within-tolerance scoring.

    Returns:
        A ``RewardBreakdown`` with individual and total scores.
    """
    gen = generalization_reward(result, tolerance)
    cal = calibration_reward(result)
    disc = discrimination_reward(result)
    val = validity_reward(source)
    itr = iteration_reward(num_tool_calls)

    total = (
        WEIGHT_GENERALIZATION * gen
        + WEIGHT_CALIBRATION * cal
        + WEIGHT_DISCRIMINATION * disc
        + WEIGHT_VALIDITY * val
        + WEIGHT_ITERATION * itr
    )
    # Normalize by total weight sum so the result is in [0, 1]
    weight_sum = (
        WEIGHT_GENERALIZATION
        + WEIGHT_CALIBRATION
        + WEIGHT_DISCRIMINATION
        + WEIGHT_VALIDITY
        + WEIGHT_ITERATION
    )
    total /= weight_sum

    return RewardBreakdown(
        generalization=gen,
        calibration=cal,
        discrimination=disc,
        validity=val,
        iteration=itr,
        total=total,
    )
