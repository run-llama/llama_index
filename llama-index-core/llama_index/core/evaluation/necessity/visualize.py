"""
Human-readable visualization for Evidence Necessity Graphs.

This module renders structured necessity graphs into
deterministic textual explanations suitable for logs,
debugging, and evaluation reports.
"""

from __future__ import annotations

from typing import List

from llama_index.core.evaluation.necessity.graph import (
    EvidenceNecessityGraph,
)


class NecessityTextRenderer:
    """
    Renders EvidenceNecessityGraph into a readable explanation.
    """

    def render(
        self,
        *,
        graph: EvidenceNecessityGraph,
        contexts: List[str],
    ) -> str:
        lines: List[str] = []

        lines.append("=== Evidence Necessity Analysis ===\n")

        for claim, node in graph.claims.items():
            lines.append(f"CLAIM: {claim}")

            if not node.supported:
                lines.append("  ❌ Unsupported / Hallucinated")
                lines.append("")
                continue

            indices = graph.necessary_contexts_for(claim)

            if not indices:
                lines.append("  ⚠️ Supported but no single necessary context")
            elif len(indices) == 1:
                idx = indices[0]
                lines.append("  ⚠️ Fragile (single-point dependency)")
                lines.append(f"    ↳ Context [{idx}]: {contexts[idx]}")
            else:
                lines.append("  ✅ Robust (multiple necessary contexts)")
                for idx in indices:
                    lines.append(f"    ↳ Context [{idx}]: {contexts[idx]}")

            lines.append("")

        redundant = graph.redundant_contexts(total_contexts=len(contexts))
        if redundant:
            lines.append("REDUNDANT CONTEXTS:")
            for idx in redundant:
                lines.append(f"  ↳ Context [{idx}]: {contexts[idx]}")
        else:
            lines.append("No redundant contexts detected.")

        return "\n".join(lines)
