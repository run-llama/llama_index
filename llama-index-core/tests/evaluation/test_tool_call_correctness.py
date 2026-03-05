"""Tests for ToolCallCorrectnessEvaluator."""

import pytest

from llama_index.core.evaluation.agent.tool_call_correctness import (
    ToolCallCorrectnessEvaluator,
)
from llama_index.core.evaluation.agent.utils import (
    compare_tool_calls,
)


# -- compare_tool_calls tests --


def test_compare_exact_match_unordered():
    expected = [
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        {"tool_name": "calculator", "tool_kwargs": {"expr": "2+2"}},
    ]
    actual = [
        {"tool_name": "calculator", "tool_kwargs": {"expr": "2+2"}},
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
    ]
    result = compare_tool_calls(expected, actual)
    assert result.matched == 2
    assert result.score == 1.0
    assert result.unmatched_expected == []
    assert result.unmatched_actual == []


def test_compare_no_match():
    expected = [{"tool_name": "search", "tool_kwargs": {"query": "weather"}}]
    actual = [{"tool_name": "calculator", "tool_kwargs": {"expr": "2+2"}}]
    result = compare_tool_calls(expected, actual)
    assert result.matched == 0
    assert result.score == 0.0
    assert len(result.unmatched_expected) == 1
    assert len(result.unmatched_actual) == 1


def test_compare_partial_match():
    expected = [
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        {"tool_name": "email", "tool_kwargs": {"to": "alice@example.com"}},
        {"tool_name": "calendar", "tool_kwargs": {"date": "2024-01-01"}},
    ]
    actual = [
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        {"tool_name": "calendar", "tool_kwargs": {"date": "2024-01-01"}},
    ]
    result = compare_tool_calls(expected, actual)
    assert result.matched == 2
    assert abs(result.score - 2 / 3) < 0.01


def test_compare_ordered_match():
    expected = [
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        {"tool_name": "calculator", "tool_kwargs": {"expr": "2+2"}},
    ]
    actual = [
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        {"tool_name": "calculator", "tool_kwargs": {"expr": "2+2"}},
    ]
    result = compare_tool_calls(expected, actual, ordered=True)
    assert result.matched == 2
    assert result.score == 1.0


def test_compare_ordered_mismatch():
    expected = [
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        {"tool_name": "calculator", "tool_kwargs": {"expr": "2+2"}},
    ]
    actual = [
        {"tool_name": "calculator", "tool_kwargs": {"expr": "2+2"}},
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
    ]
    result = compare_tool_calls(expected, actual, ordered=True)
    assert result.matched == 0
    assert result.score == 0.0


def test_compare_subset_kwargs():
    expected = [{"tool_name": "search", "tool_kwargs": {"query": "weather"}}]
    actual = [
        {
            "tool_name": "search",
            "tool_kwargs": {"query": "weather", "limit": 10, "lang": "en"},
        }
    ]
    result = compare_tool_calls(expected, actual, strict_kwargs=False)
    assert result.matched == 1
    assert result.score == 1.0


def test_compare_strict_kwargs():
    expected = [{"tool_name": "search", "tool_kwargs": {"query": "weather"}}]
    actual = [
        {
            "tool_name": "search",
            "tool_kwargs": {"query": "weather", "limit": 10},
        }
    ]
    result = compare_tool_calls(expected, actual, strict_kwargs=True)
    assert result.matched == 0
    assert result.score == 0.0


def test_compare_empty_expected():
    result = compare_tool_calls([], [{"tool_name": "search", "tool_kwargs": {}}])
    assert result.score == 1.0
    assert result.matched == 0
    assert result.total_expected == 0


def test_compare_empty_actual():
    result = compare_tool_calls([{"tool_name": "search", "tool_kwargs": {}}], [])
    assert result.score == 0.0
    assert result.matched == 0


def test_compare_both_empty():
    result = compare_tool_calls([], [])
    assert result.score == 1.0


def test_compare_tool_id_ignored():
    """tool_id should be ignored during comparison."""
    expected = [
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}, "tool_id": "abc"}
    ]
    actual = [
        {"tool_name": "search", "tool_kwargs": {"query": "weather"}, "tool_id": "xyz"}
    ]
    result = compare_tool_calls(expected, actual)
    assert result.matched == 1
    assert result.score == 1.0


# -- ToolCallCorrectnessEvaluator tests --


@pytest.mark.asyncio
async def test_evaluator_exact_match():
    evaluator = ToolCallCorrectnessEvaluator()
    result = await evaluator.aevaluate(
        expected_tool_calls=[
            {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        ],
        actual_tool_calls=[
            {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        ],
    )
    assert result.score == 1.0
    assert result.passing is True


@pytest.mark.asyncio
async def test_evaluator_no_match():
    evaluator = ToolCallCorrectnessEvaluator()
    result = await evaluator.aevaluate(
        expected_tool_calls=[
            {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        ],
        actual_tool_calls=[
            {"tool_name": "calculator", "tool_kwargs": {"expr": "1+1"}},
        ],
    )
    assert result.score == 0.0
    assert result.passing is False
    assert "Missing expected" in result.feedback


@pytest.mark.asyncio
async def test_evaluator_partial_match():
    evaluator = ToolCallCorrectnessEvaluator(threshold=0.5)
    result = await evaluator.aevaluate(
        expected_tool_calls=[
            {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
            {"tool_name": "email", "tool_kwargs": {"to": "alice@example.com"}},
        ],
        actual_tool_calls=[
            {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        ],
    )
    assert result.score == 0.5
    assert result.passing is True  # threshold=0.5


@pytest.mark.asyncio
async def test_evaluator_custom_threshold_fail():
    evaluator = ToolCallCorrectnessEvaluator(threshold=0.8)
    result = await evaluator.aevaluate(
        expected_tool_calls=[
            {"tool_name": "a", "tool_kwargs": {}},
            {"tool_name": "b", "tool_kwargs": {}},
            {"tool_name": "c", "tool_kwargs": {}},
        ],
        actual_tool_calls=[
            {"tool_name": "a", "tool_kwargs": {}},
            {"tool_name": "b", "tool_kwargs": {}},
        ],
    )
    assert abs(result.score - 2 / 3) < 0.01
    assert result.passing is False  # 0.667 < 0.8


@pytest.mark.asyncio
async def test_evaluator_missing_args_raises():
    evaluator = ToolCallCorrectnessEvaluator()
    with pytest.raises(ValueError, match="must be provided"):
        await evaluator.aevaluate(
            expected_tool_calls=[{"tool_name": "search", "tool_kwargs": {}}],
            actual_tool_calls=None,
        )


@pytest.mark.asyncio
async def test_evaluator_ordered():
    evaluator = ToolCallCorrectnessEvaluator(ordered=True)
    result = await evaluator.aevaluate(
        expected_tool_calls=[
            {"tool_name": "a", "tool_kwargs": {}},
            {"tool_name": "b", "tool_kwargs": {}},
        ],
        actual_tool_calls=[
            {"tool_name": "b", "tool_kwargs": {}},
            {"tool_name": "a", "tool_kwargs": {}},
        ],
    )
    assert result.score == 0.0
    assert result.passing is False


@pytest.mark.asyncio
async def test_evaluator_feedback_content():
    evaluator = ToolCallCorrectnessEvaluator()
    result = await evaluator.aevaluate(
        expected_tool_calls=[
            {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        ],
        actual_tool_calls=[
            {"tool_name": "search", "tool_kwargs": {"query": "weather"}},
        ],
    )
    assert "Matched 1/1" in result.feedback
