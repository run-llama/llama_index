"""Integration tests for agent evaluation (requires real OpenAI API key).

Run with: pytest test_agent_eval_integration.py -v -m integration
"""

import os

import pytest

from llama_index.core.evaluation.agent.goal_success import (
    AgentGoalSuccessEvaluator,
)
from llama_index.core.evaluation.agent.tool_call_correctness import (
    ToolCallCorrectnessEvaluator,
)

skip_no_api_key = pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not set",
)


@skip_no_api_key
@pytest.mark.integration()
def test_tool_call_real_scenario():
    """Realistic tool call comparison with multiple tools."""
    evaluator = ToolCallCorrectnessEvaluator(threshold=0.5)

    expected = [
        {"tool_name": "web_search", "tool_kwargs": {"query": "San Francisco weather"}},
        {"tool_name": "format_response", "tool_kwargs": {"format": "brief"}},
    ]
    actual = [
        {"tool_name": "web_search", "tool_kwargs": {"query": "San Francisco weather today", "limit": 5}},
        {"tool_name": "format_response", "tool_kwargs": {"format": "brief"}},
    ]
    result = evaluator.evaluate(
        expected_tool_calls=expected,
        actual_tool_calls=actual,
    )
    # web_search: subset kwargs match (query matches, limit is extra)
    # format_response: exact match
    assert result.score >= 0.5
    assert result.passing is True
    assert result.feedback is not None


@skip_no_api_key
@pytest.mark.integration()
def test_goal_success_real_llm():
    """Test AgentGoalSuccessEvaluator with a real OpenAI call."""
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    evaluator = AgentGoalSuccessEvaluator(llm=llm)

    result = evaluator.evaluate(
        query="What is the capital of France?",
        response="The capital of France is Paris.",
        reference="Paris",
    )
    assert result.score is not None
    assert 1.0 <= result.score <= 5.0
    assert result.passing is True  # Should score high
    assert result.feedback is not None
    print(f"Score: {result.score}, Feedback: {result.feedback}")


@skip_no_api_key
@pytest.mark.integration()
def test_goal_success_with_tool_history():
    """Full agent scenario with tool call history."""
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    evaluator = AgentGoalSuccessEvaluator(llm=llm)

    result = evaluator.evaluate(
        query="Find cheap flights from NYC to London for next week",
        response="I found 3 flights. The cheapest is Delta for $450 departing Monday.",
        contexts=[
            "Called flight_search(from='NYC', to='London', date_range='next week') -> [Delta $450 Mon, United $520 Tue, BA $610 Wed]",
            "Called sort_results(by='price', order='asc') -> [Delta $450, United $520, BA $610]",
        ],
        reference="Find and present flight options sorted by price",
    )
    assert result.score is not None
    assert 1.0 <= result.score <= 5.0
    assert result.feedback is not None
    print(f"Score: {result.score}, Feedback: {result.feedback}")
