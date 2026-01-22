"""
Claim-Conditioned Necessity Analyzer.

Determines which evidence contexts are strictly necessary for a claim
using deterministic counterfactual intervention.

────────────────────────────────────────────────────────────────────
HARD GUARANTEES
────────────────────────────────────────────────────────────────────
- Deterministic execution
- Fail-closed semantics
- O(N) oracle calls per claim (exactly N + 1)
- Bounded concurrency
- Explicit observability hooks

No probabilistic assumptions are made.
"""

from __future__ import annotations

import asyncio
from typing import Sequence, List, Optional
from dataclasses import dataclass

from llama_index.core.evaluation.necessity.oracle import (
    ClaimSupportOracle,
    ClaimSupportResult,
)


# ---------------------------------------------------------------------
# Public result contract
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class NecessityResult:
    """
    Result of necessity analysis for a single claim.
    """
    claim: str
    necessary_context_indices: List[int]
    initially_supported: bool


# ---------------------------------------------------------------------
# Analyzer implementation
# ---------------------------------------------------------------------

class ClaimNecessityAnalyzer:
    """
    Deterministic analyzer for claim-conditioned evidence necessity.

    A context C_i is considered NECESSARY for claim Q iff:
        1. Q is supported with all contexts present
        2. Removing C_i (while leaving at least one context)
           makes Q unsupported
    """

    def __init__(
        self,
        *,
        oracle: ClaimSupportOracle,
        max_contexts: Optional[int] = None,
        chunk_size: Optional[int] = None,
        parallelism: int = 1,
        observer=None,
    ) -> None:
        self._oracle = oracle
        self._max_contexts = max_contexts
        self._chunk_size = chunk_size
        self._parallelism = parallelism
        self._observer = observer

        self._semaphore = asyncio.Semaphore(parallelism)

    def _emit(self, event: str, **data) -> None:
        if self._observer:
            self._observer({"event": event, **data})

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def analyze(
        self,
        *,
        claim: str,
        contexts: Sequence[str],
    ) -> NecessityResult:
        """
        Analyze which contexts are necessary for the given claim.
        """

        # -------------------------------------------------------------
        # Safeguard: max_contexts
        # -------------------------------------------------------------

        if (
            self._max_contexts is not None
            and len(contexts) > self._max_contexts
        ):
            raise ValueError("max_contexts exceeded")

        # -------------------------------------------------------------
        # Step 1: Baseline support check (exactly once)
        # -------------------------------------------------------------

        async with self._semaphore:
            baseline: ClaimSupportResult = await self._oracle.check(
                claim=claim,
                contexts=contexts,
            )

        self._emit("baseline_checked")

        if not baseline.supported:
            return NecessityResult(
                claim=claim,
                necessary_context_indices=[],
                initially_supported=False,
            )

        # -------------------------------------------------------------
        # Step 2: Counterfactual removals (O(N), bounded)
        # -------------------------------------------------------------

        necessary: List[int] = []

        indices = range(len(contexts))
        chunks = (
            self._chunk(indices)
            if self._chunk_size
            else [list(indices)]
        )

        for chunk in chunks:
            tasks = [
                self._check_removal(
                    claim=claim,
                    contexts=contexts,
                    remove_index=i,
                )
                for i in chunk
            ]

            results = await asyncio.gather(*tasks)

            for idx, supported in results:
                if not supported:
                    necessary.append(idx)

        self._emit("analysis_completed")

        return NecessityResult(
            claim=claim,
            necessary_context_indices=necessary,
            initially_supported=True,
        )

    # -----------------------------------------------------------------
    # Internal execution
    # -----------------------------------------------------------------

    async def _check_removal(
        self,
        *,
        claim: str,
        contexts: Sequence[str],
        remove_index: int,
    ) -> tuple[int, bool]:
        """
        Check claim support after removing a single context.

        IMPORTANT:
        - Empty-context collapse is ignored
        - Necessity is inferred only when other evidence remains
        """
        reduced_contexts = [
            ctx
            for i, ctx in enumerate(contexts)
            if i != remove_index
        ]

        # 🚫 Do NOT infer necessity from empty-context collapse
        if not reduced_contexts:
            return remove_index, True

        async with self._semaphore:
            result = await self._oracle.check(
                claim=claim,
                contexts=reduced_contexts,
            )

        return remove_index, result.supported

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def _chunk(self, indices):
        """
        Yield chunks of indices according to chunk_size.
        """
        chunk: List[int] = []
        for idx in indices:
            chunk.append(idx)
            if len(chunk) == self._chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
