"""Runner protocol for evaluating candidate rubric functions."""

from __future__ import annotations

from typing import List, Optional, Protocol

from environments.rubric_discovery.env.types import EvaluationResult, LabeledExample


class EvaluationRunner(Protocol):
    """Protocol for running a candidate rubric_fn against labeled examples."""

    def evaluate(
        self,
        source: str,
        examples: List[LabeledExample],
        timeout_s: int = 10,
    ) -> EvaluationResult:
        """Execute the candidate rubric and return an EvaluationResult.

        Args:
            source: Python source code containing a ``rubric_fn`` definition.
            examples: List of labeled examples to evaluate against.
            timeout_s: Maximum execution time in seconds.

        Returns:
            An ``EvaluationResult`` containing predictions and validity info.
        """
        ...
