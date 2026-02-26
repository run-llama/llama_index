"""Rubric construction for the rubric-discovery environment.

Builds a ``Rubric`` (or equivalent scoring object) that evaluates
a model's final rubric_fn against held-out test examples.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from environments.rubric_discovery.env.candidate.extractor import (
    extract_rubric_fn_source,
)
from environments.rubric_discovery.env.evaluation.service import EvaluationService
from environments.rubric_discovery.env.runtime.rewards import (
    RewardBreakdown,
    compute_rewards,
)
from environments.rubric_discovery.env.types import (
    EvaluationResult,
    LabeledExample,
    RubricDiscoveryConfig,
)


class RubricScorer:
    """Scores a model's final output by evaluating its rubric_fn on held-out data.

    This is the core scoring component that computes the multi-dimensional
    reward signal used for RL training.
    """

    def __init__(
        self,
        config: RubricDiscoveryConfig,
        test_examples: List[LabeledExample],
        eval_service: EvaluationService,
    ) -> None:
        self._config = config
        self._test_examples = test_examples
        self._eval_service = eval_service

    def score(
        self,
        model_output: str,
        num_tool_calls: int = 0,
    ) -> Dict[str, Any]:
        """Score the model's final output.

        Args:
            model_output: The model's complete output, from which we extract
                the final ``rubric_fn`` definition.
            num_tool_calls: Total tool calls made during the episode.

        Returns:
            A dictionary with reward breakdown and metrics.
        """
        source = extract_rubric_fn_source(model_output)

        # Evaluate on held-out test examples
        if source is not None:
            result = self._eval_service.evaluate(
                source,
                self._test_examples,
                timeout_s=self._config.eval_timeout_s,
            )
        else:
            result = EvaluationResult(
                labels=[e.score for e in self._test_examples],
                valid=False,
                error="No rubric_fn found in model output.",
            )

        # Compute rewards
        rewards = compute_rewards(
            result=result,
            source=source,
            num_tool_calls=num_tool_calls,
            tolerance=self._config.eval_tolerance,
        )

        # Compute primary metric
        within_tolerance_rate = 0.0
        if result.valid and result.predictions:
            correct = sum(
                1
                for p, l in zip(result.predictions, result.labels)
                if abs(p - l) < self._config.eval_tolerance
            )
            within_tolerance_rate = correct / len(result.labels)

        return {
            "reward": rewards.total,
            "rewards": rewards.to_dict(),
            "metrics": {
                "within_tolerance_rate": within_tolerance_rate,
                "accuracy": within_tolerance_rate,  # backwards compat alias
                "valid": result.valid,
                "num_test_examples": len(self._test_examples),
                "num_tool_calls": num_tool_calls,
            },
            "source": source,
            "error": result.error,
        }


def build_rubric(
    config: RubricDiscoveryConfig,
    test_examples: List[LabeledExample],
    eval_service: EvaluationService,
) -> RubricScorer:
    """Factory function to create a RubricScorer.

    This is the main entry point used by the environment's
    ``load_environment`` function.

    Args:
        config: Environment configuration.
        test_examples: Held-out test examples for evaluation.
        eval_service: The evaluation service instance.

    Returns:
        A ``RubricScorer`` ready to score model outputs.
    """
    return RubricScorer(
        config=config,
        test_examples=test_examples,
        eval_service=eval_service,
    )
