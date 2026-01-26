"""
Evidence Necessity Graph (ENG).

Represents claim-to-evidence necessity relations discovered via
deterministic counterfactual analysis.

────────────────────────────────────────────────────────────────────
DESIGN INTENT
────────────────────────────────────────────────────────────────────
- Structural (not scalar, not probabilistic)
- Human-legible
- Machine-consumable
- Versioned & extension-safe

This graph is a PURE artifact:
- No inference
- No scoring
- No re-evaluation
"""

from __future__ import annotations

from typing import Dict, List, ClassVar
from dataclasses import dataclass, field

from llama_index.core.evaluation.necessity.necessity import NecessityResult


# ---------------------------------------------------------------------
# Public data contracts
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimNode:
    """
    Node representing a factual claim.
    """
    text: str
    supported: bool


@dataclass(frozen=True)
class ContextNode:
    """
    Node representing an evidence context.
    """
    index: int


@dataclass
class EvidenceNecessityGraph:
    """
    Bipartite graph connecting claims to necessary contexts.

    Structural invariants:
        - Claims are keyed by claim text
        - Edges map claim -> list[int]
        - Context indices are opaque identifiers
    """

    # -----------------------------------------------------------------
    # Schema versioning
    # -----------------------------------------------------------------

    SCHEMA_VERSION: ClassVar[int] = 1

    claims: Dict[str, ClaimNode] = field(default_factory=dict)
    edges: Dict[str, List[int]] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def add_result(self, result: NecessityResult) -> None:
        """
        Add a NecessityResult to the graph.

        Idempotent per claim: later results overwrite earlier ones.
        """
        self.claims[result.claim] = ClaimNode(
            text=result.claim,
            supported=result.initially_supported,
        )

        self.edges[result.claim] = list(
            result.necessary_context_indices
        )

    # -----------------------------------------------------------------
    # Structural queries
    # -----------------------------------------------------------------

    def unsupported_claims(self) -> List[str]:
        """
        Claims unsupported even with full evidence.
        """
        return [
            claim
            for claim, node in self.claims.items()
            if not node.supported
        ]

    def necessary_contexts_for(self, claim: str) -> List[int]:
        """
        Necessary context indices for a given claim.
        """
        return list(self.edges.get(claim, []))

    def all_necessary_contexts(self) -> List[int]:
        """
        Sorted list of all context indices required by at least one claim.
        """
        unique = set()
        for indices in self.edges.values():
            unique.update(indices)
        return sorted(unique)

    # -----------------------------------------------------------------
    # Redundancy & robustness analysis
    # -----------------------------------------------------------------

    def fragile_claims(self) -> List[str]:
        """
        Claims dependent on exactly one necessary context.
        """
        return [
            claim
            for claim, indices in self.edges.items()
            if len(indices) == 1
        ]

    def robust_claims(self) -> List[str]:
        """
        Claims with redundant evidence (multiple necessary contexts).
        """
        return [
            claim
            for claim, indices in self.edges.items()
            if len(indices) > 1
        ]

    def redundant_contexts(self, total_contexts: int) -> List[int]:
        """
        Context indices unused by any claim.
        """
        used = set(self.all_necessary_contexts())
        return [
            i for i in range(total_contexts)
            if i not in used
        ]

    # -----------------------------------------------------------------
    # Serialization (round-trip safe, defensive)
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Dict]:
        """
        Serialize graph to a versioned dictionary.
        """
        return {
            "version": self.SCHEMA_VERSION,
            "claims": {
                claim: {
                    "supported": node.supported,
                    "necessary_context_indices": list(
                        self.edges.get(claim, [])
                    ),
                }
                for claim, node in self.claims.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "EvidenceNecessityGraph":
        """
        Deserialize a graph from a dictionary.

        Defensive by design: malformed payloads fail fast.
        """

        # -------------------------------------------------------------
        # Top-level validation
        # -------------------------------------------------------------

        if not isinstance(payload, dict):
            raise ValueError("Invalid payload type")

        if payload.get("version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported schema version")

        claims = payload.get("claims")
        if not isinstance(claims, dict):
            raise ValueError("Invalid claims structure")

        # -------------------------------------------------------------
        # Payload decoding
        # -------------------------------------------------------------

        graph = cls()

        for claim, data in claims.items():
            if not isinstance(claim, str):
                raise ValueError("Invalid claim key")

            if not isinstance(data, dict):
                raise ValueError("Invalid claim payload")

            supported = data.get("supported")
            if not isinstance(supported, bool):
                raise ValueError("Invalid supported flag")

            indices = data.get("necessary_context_indices")
            if not isinstance(indices, list):
                raise ValueError("Invalid context indices")

            # Ensure indices are integers
            for idx in indices:
                if not isinstance(idx, int):
                    raise ValueError("Invalid context index type")

            graph.claims[claim] = ClaimNode(
                text=claim,
                supported=supported,
            )
            graph.edges[claim] = list(indices)

        return graph
