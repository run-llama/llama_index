"""High-level evaluation service orchestrating backend execution and metrics."""

from __future__ import annotations

from typing import Dict, List, Optional

from environments.rubric_discovery.env.candidate.extractor import (
    compile_rubric_fn,
    extract_rubric_fn_source,
    probe_callability,
    validate_signature,
)
from environments.rubric_discovery.env.evaluation.backends import (
    BaseBackend,
    SubprocessBackend,
    resolve_backend,
)
from environments.rubric_discovery.env.types import (
    EvaluationResult,
    ExecutionBackend,
    LabeledExample,
    RubricDiscoveryConfig,
)


class EvaluationService:
    """Orchestrates candidate rubric_fn extraction, validation, and evaluation.

    This service is the main entry point for the tool wrappers
    (``test_rubric``, ``score_examples``) in the runtime layer.
    """

    def __init__(self, config: RubricDiscoveryConfig) -> None:
        self._config = config
        self._backend: BaseBackend = resolve_backend(
            config.get_execution_backend()
        )

    @property
    def tolerance(self) -> float:
        """Tolerance for within-tolerance scoring."""
        return self._config.eval_tolerance

    def extract_source(self, text: str) -> Optional[str]:
        """Extract ``rubric_fn`` source from model output text."""
        return extract_rubric_fn_source(text)

    def validate(self, source: str) -> Dict[str, object]:
        """Validate source code: signature check + callability probe.

        Returns a dict with ``valid``, ``signature_ok``, ``callable_ok``,
        and optional ``error`` fields.
        """
        sig_ok, sig_err = validate_signature(source)
        if not sig_ok:
            return {
                "valid": False,
                "signature_ok": False,
                "callable_ok": False,
                "error": sig_err,
            }

        call_ok, call_err = probe_callability(source)
        return {
            "valid": call_ok,
            "signature_ok": True,
            "callable_ok": call_ok,
            "error": call_err,
        }

    def evaluate(
        self,
        source: str,
        examples: List[LabeledExample],
        timeout_s: Optional[int] = None,
    ) -> EvaluationResult:
        """Run the candidate rubric against *examples*.

        Args:
            source: Python source defining ``rubric_fn``.
            examples: Labeled examples to score.
            timeout_s: Override for execution timeout.

        Returns:
            An ``EvaluationResult`` with predictions and labels.
        """
        t = timeout_s or self._config.eval_timeout_s
        return self._backend.execute(source, examples, timeout_s=t)

    def score_within_tolerance(
        self,
        result: EvaluationResult,
        tolerance: Optional[float] = None,
    ) -> float:
        """Compute the fraction of predictions within tolerance of labels.

        Args:
            result: An evaluation result with predictions and labels.
            tolerance: Override the configured tolerance.

        Returns:
            Fraction in ``[0.0, 1.0]``.
        """
        tol = tolerance if tolerance is not None else self.tolerance
        if not result.valid or not result.predictions:
            return 0.0

        correct = sum(
            1
            for pred, label in zip(result.predictions, result.labels)
            if abs(pred - label) < tol
        )
        return correct / len(result.labels)
