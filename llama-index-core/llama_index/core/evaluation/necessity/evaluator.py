from __future__ import annotations
import json
from typing import Any, Optional, Sequence
"""
Evidence Necessity Evaluator (ENE).

Constructs an Evidence Necessity Graph (ENG) by performing
deterministic, claim-conditioned counterfactual analysis.

Pipeline:
    1. Extract factual claims from the response
    2. Verify baseline claim support
    3. Identify necessary evidence contexts
    4. Assemble a structural Evidence Necessity Graph

────────────────────────────────────────────────────────────────────
DESIGN PRINCIPLES
────────────────────────────────────────────────────────────────────
- Structural output (graph), not scalar scoring
- Deterministic and evaluation-safe
- Fail-closed semantics
- Extensible without schema breakage

This evaluator does NOT:
- Assign confidence scores
- Rank claims
- Re-evaluate model reasoning
"""


from llama_index.core.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
)

from llama_index.core.evaluation.necessity.claims import extract_claims
from llama_index.core.evaluation.necessity.oracle import ClaimSupportOracle
from llama_index.core.evaluation.necessity.necessity import (
    ClaimNecessityAnalyzer,
)
from llama_index.core.evaluation.necessity.graph import (
    EvidenceNecessityGraph,
)


class EvidenceNecessityEvaluator(BaseEvaluator):
    """
    Evaluator that builds an Evidence Necessity Graph (ENG).

    Output semantics:
        - score: None (structure > scalar)
        - feedback: serialized, versioned ENG

    This evaluator is intentionally non-opinionated:
        it exposes structure, not judgment.
    """

    def __init__(
        self,
        *,
        oracle: Optional[ClaimSupportOracle] = None,
    ) -> None:
        """
        Args:
            oracle:
                Deterministic oracle used for claim support verification.
                If omitted, a default evaluation-safe oracle is constructed.
        """
        self._oracle = oracle or ClaimSupportOracle()
        self._analyzer = ClaimNecessityAnalyzer(
            oracle=self._oracle,
        )
        # -----------------------------------------------------------------
    # BaseEvaluator abstract method stubs
    # -----------------------------------------------------------------

    def _get_prompts(self):
        return {}

    def _update_prompts(self, prompts):
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> EvaluationResult:
        """
        Execute evidence necessity evaluation.

        Preconditions:
            - response must be provided
            - contexts must be provided

        Returns:
            EvaluationResult with serialized ENG as feedback.
        """

        # -------------------------------------------------------------
        # Input validation (fail fast, fail closed)
        # -------------------------------------------------------------

        if response is None or contexts is None:
            return EvaluationResult(
                invalid_result=True,
                invalid_reason=(
                    "EvidenceNecessityEvaluator requires both "
                    "`response` and `contexts`."
                ),
            )

        # -------------------------------------------------------------
        # Step 1: Claim extraction
        # -------------------------------------------------------------

        claims = extract_claims(response)

        graph = EvidenceNecessityGraph()

        # -------------------------------------------------------------
        # Step 2: Per-claim necessity analysis
        # -------------------------------------------------------------

        for claim in claims:
            result = await self._analyzer.analyze(
                claim=claim,
                contexts=contexts,
            )
            graph.add_result(result)

        # -------------------------------------------------------------
        # Step 3: Return structural evaluation artifact
        # -------------------------------------------------------------

        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            score=None,  # structural evaluation only
            feedback=json.dumps(graph.to_dict()),
        )
