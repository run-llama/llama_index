"""
Claim extraction for Evidence Necessity Evaluation.

Deterministic, sentence-level extraction.
No LLM dependency.
"""

from __future__ import annotations
from typing import List
import re


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_claims(answer: str) -> List[str]:
    """
    Extract atomic claims from an answer.

    Strategy:
    - Sentence-level segmentation
    - Deterministic
    - Conservative (no semantic splitting)
    """
    if not answer:
        return []

    parts = _SENTENCE_SPLIT.split(answer.strip())
    return [p.strip() for p in parts if p.strip()]
