"""
Field normalisation for Azure Document Intelligence KV output.

Azure DI returns raw key strings that contain:
  - Trailing spaces          (e.g. ``"Wages/Salary/Tips - HHA "``)
  - Known field-name typos   (e.g. ``"SeconD Read"``, ``"Student Conrtirbution"``)
  - ``>`` characters in keys (e.g. ``"Based Year Academic Expenses> Tuition Paid"``)
  - Values encoded as quoted strings that should be numeric
    (e.g. ``'"75000"'`` → ``"75000"``)

This module provides:
  - :func:`normalize_key`   — clean a single field key
  - :func:`normalize_value` — clean a single field value
  - :func:`normalize_pairs` — normalise a full list of (key, value, confidence) tuples
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Known key corrections — maps the raw Azure DI string to the canonical name.
# Source: production observation on PowerFAIDS / OnBase documents.
# ---------------------------------------------------------------------------
_KEY_CORRECTIONS: dict[str, str] = {
    # PowerFAIDS summary field typos
    "SeconD Read":                      "Second Read",
    "Student Conrtirbution":            "Student Contribution",
    # Trailing-space variants (stripped automatically, listed for reference)
    "Wages/Salary/Tips - HHA ":         "Wages/Salary/Tips - HHA",
    "Wages/Salary/Tips - HHB ":         "Wages/Salary/Tips - HHB",
    # '>' separator in long key names
    "Based Year Academic Expenses> Tuition Paid - HHA": (
        "Based Year Academic Expenses - Tuition Paid - HHA"
    ),
    "Based Year Academic Expenses> Tuition Paid - HHB": (
        "Based Year Academic Expenses - Tuition Paid - HHB"
    ),
}

# Keys whose values arrive as quoted strings in JSON (e.g. ``'"75000"'``).
_QUOTED_NUMERIC_KEYS: frozenset[str] = frozenset(
    {
        "IM Original Need",
        "Adjusted Gross Income",
        "Total Income",
        "Wages, salaries, tips, etc.",
    }
)


def normalize_key(raw_key: Optional[str]) -> Optional[str]:
    """
    Normalise a raw Azure DI key string.

    Steps applied in order:
    1. Strip leading/trailing whitespace.
    2. Replace ``>`` with ``-`` (separator normalisation).
    3. Apply known correction map.

    Returns ``None`` when the input is ``None`` or blank after stripping.
    """
    if not raw_key:
        return None
    key = raw_key.strip()
    if not key:
        return None
    # Replace '>' separator — normalise to ' - ' and collapse any double spaces
    import re as _re
    key = _re.sub(r'\s*>\s*', ' - ', key).strip()
    # Apply known corrections (check both the raw and the '>' -cleaned variant)
    return _KEY_CORRECTIONS.get(key, key)


def normalize_value(raw_value: Optional[str], key: Optional[str] = None) -> Optional[str]:
    """
    Normalise a raw Azure DI value string.

    Steps:
    1. Strip leading/trailing whitespace.
    2. For keys in ``_QUOTED_NUMERIC_KEYS``, strip surrounding double-quotes
       so the caller can safely parse the value as a number.

    Returns ``None`` when the input is ``None`` or blank.
    """
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    # Strip surrounding quotes for known numeric fields
    if key and key in _QUOTED_NUMERIC_KEYS:
        value = value.strip('"')
    return value


def normalize_pairs(
    pairs: list[tuple[Optional[str], Optional[str], Optional[float]]],
) -> list[tuple[str, Optional[str], Optional[float]]]:
    """
    Normalise a list of ``(key, value, confidence)`` tuples.

    Pairs with a blank key after normalisation are dropped.

    Returns a list of ``(normalised_key, normalised_value, confidence)`` tuples.
    """
    result = []
    for raw_key, raw_value, confidence in pairs:
        key = normalize_key(raw_key)
        if not key:
            continue
        value = normalize_value(raw_value, key)
        result.append((key, value, confidence))
    return result
