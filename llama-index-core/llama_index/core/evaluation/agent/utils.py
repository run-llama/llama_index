"""Utility functions for agent evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolCallComparisonResult:
    """Result of comparing expected vs actual tool calls."""

    matched: int = 0
    total_expected: int = 0
    total_actual: int = 0
    unmatched_expected: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_actual: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def score(self) -> float:
        if self.total_expected == 0:
            return 1.0
        return self.matched / self.total_expected


def _kwargs_match(
    expected_kwargs: Dict[str, Any],
    actual_kwargs: Dict[str, Any],
    strict: bool = False,
) -> bool:
    """
    Check if expected kwargs match actual kwargs.

    Args:
        expected_kwargs: The expected keyword arguments.
        actual_kwargs: The actual keyword arguments.
        strict: If True, kwargs must match exactly.
            If False, expected kwargs must be a subset of actual kwargs.

    """
    if strict:
        return expected_kwargs == actual_kwargs
    return all(
        k in actual_kwargs and actual_kwargs[k] == v for k, v in expected_kwargs.items()
    )


def compare_tool_calls(
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
    ordered: bool = False,
    strict_kwargs: bool = False,
) -> ToolCallComparisonResult:
    """
    Compare expected tool calls against actual tool calls.

    Each tool call is a dict with keys:
        - tool_name (str): Name of the tool.
        - tool_kwargs (dict): Keyword arguments passed to the tool.

    The tool_id field is ignored if present (auto-generated IDs
    never match between expected and actual).

    Args:
        expected: List of expected tool call dicts.
        actual: List of actual tool call dicts.
        ordered: If True, compare position-by-position.
            If False, use greedy unordered matching.
        strict_kwargs: If True, kwargs must match exactly.
            If False, expected kwargs must be a subset of actual.

    Returns:
        ToolCallComparisonResult with match counts and unmatched lists.

    """
    result = ToolCallComparisonResult(
        total_expected=len(expected),
        total_actual=len(actual),
    )

    if ordered:
        for i, exp in enumerate(expected):
            if i < len(actual):
                act = actual[i]
                if exp.get("tool_name") == act.get("tool_name") and _kwargs_match(
                    exp.get("tool_kwargs", {}),
                    act.get("tool_kwargs", {}),
                    strict=strict_kwargs,
                ):
                    result.matched += 1
                else:
                    result.unmatched_expected.append(exp)
            else:
                result.unmatched_expected.append(exp)
        # Any extra actual calls beyond expected length
        for act in actual[len(expected) :]:
            result.unmatched_actual.append(act)
    else:
        # Greedy unordered matching
        remaining_actual = list(actual)
        for exp in expected:
            found = False
            for j, act in enumerate(remaining_actual):
                if exp.get("tool_name") == act.get("tool_name") and _kwargs_match(
                    exp.get("tool_kwargs", {}),
                    act.get("tool_kwargs", {}),
                    strict=strict_kwargs,
                ):
                    result.matched += 1
                    remaining_actual.pop(j)
                    found = True
                    break
            if not found:
                result.unmatched_expected.append(exp)
        result.unmatched_actual = remaining_actual

    return result
