"""
Dataset runner for Evidence Necessity Evaluation.

This module executes the evaluator over structured datasets
without assuming any specific benchmark format.
"""

from __future__ import annotations

import json
from typing import Iterable, Dict, Any, List
from dataclasses import dataclass

from llama_index.core.evaluation.necessity.evaluator import (
    EvidenceNecessityEvaluator,
)


# ---------------------------------------------------------------------
# Public data contracts
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetExample:
    """
    Single evaluation example.
    """
    query: str
    response: str
    contexts: List[str]


@dataclass
class DatasetResult:
    """
    Aggregated dataset-level result.
    """
    examples: int
    unsupported_claims: int
    fragile_claims: int
    robust_claims: int
    redundant_contexts: int
    raw_outputs: List[Dict[str, Any]]


# ---------------------------------------------------------------------
# Runner implementation
# ---------------------------------------------------------------------

class NecessityDatasetRunner:
    """
    Executes EvidenceNecessityEvaluator over a dataset.
    """

    def __init__(
        self,
        *,
        evaluator: EvidenceNecessityEvaluator,
    ) -> None:
        self._evaluator = evaluator

    async def run(
        self,
        dataset: Iterable[DatasetExample],
    ) -> DatasetResult:
        raw_outputs: List[Dict[str, Any]] = []

        unsupported = 0
        fragile = 0
        robust = 0
        redundant = 0
        examples = 0

        for example in dataset:
            result = await self._evaluator.aevaluate(
                query=example.query,
                response=example.response,
                contexts=example.contexts,
            )
            
            graph = json.loads(result.feedback)
            claims = graph.get("claims", {})

            examples += 1

            for _, data in claims.items():
                if not data["supported"]:
                    unsupported += 1
                elif len(data["necessary_context_indices"]) == 1:
                    fragile += 1
                elif len(data["necessary_context_indices"]) > 1:
                    robust += 1

            used_contexts = {
                idx
                for data in claims.values()
                for idx in data["necessary_context_indices"]
            }

            redundant += max(
                0, len(example.contexts) - len(used_contexts)
            )

            raw_outputs.append(
                {
                    "query": example.query,
                    "claims": claims,
                }
            )

        return DatasetResult(
            examples=examples,
            unsupported_claims=unsupported,
            fragile_claims=fragile,
            robust_claims=robust,
            redundant_contexts=redundant,
            raw_outputs=raw_outputs,
        )
