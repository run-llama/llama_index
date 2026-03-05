"""Tests for AgentGoalSuccessEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.evaluation.agent.goal_success import (
    AgentGoalSuccessEvaluator,
)


def _make_mock_llm(response_text: str) -> MagicMock:
    """Create a mock LLM that returns the given text from apredict."""
    mock_llm = MagicMock()
    mock_llm.apredict = AsyncMock(return_value=response_text)
    return mock_llm


@pytest.mark.asyncio()
async def test_goal_success_high_score():
    mock_llm = _make_mock_llm("5.0\nThe agent perfectly achieved the goal.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    result = await evaluator.aevaluate(
        query="Find the weather in San Francisco",
        response="The weather in San Francisco is 65F and sunny.",
    )
    assert result.score == 5.0
    assert result.passing is True
    assert "perfectly" in result.feedback


@pytest.mark.asyncio()
async def test_goal_success_low_score():
    mock_llm = _make_mock_llm("2.0\nThe agent failed to complete the task.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    result = await evaluator.aevaluate(
        query="Book a flight to New York",
        response="I found some hotels in Boston.",
    )
    assert result.score == 2.0
    assert result.passing is False


@pytest.mark.asyncio()
async def test_goal_success_with_reference():
    mock_llm = _make_mock_llm("4.0\nClose to the expected outcome.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    result = await evaluator.aevaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        reference="The capital of France is Paris.",
    )
    assert result.score == 4.0
    assert result.passing is True

    # Verify the LLM was called with the reference
    call_kwargs = mock_llm.apredict.call_args
    assert "Paris" in str(call_kwargs)


@pytest.mark.asyncio()
async def test_goal_success_with_contexts():
    mock_llm = _make_mock_llm("5.0\nCorrect tool usage and response.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    result = await evaluator.aevaluate(
        query="Find the weather in San Francisco",
        response="The weather is 65F and sunny.",
        contexts=[
            "Called weather_api(city='San Francisco') -> '65F, sunny'",
            "Called format_response(temp='65F', conditions='sunny')",
        ],
    )
    assert result.score == 5.0
    assert result.passing is True

    # Verify tool history was passed to LLM
    call_kwargs = mock_llm.apredict.call_args
    assert "weather_api" in str(call_kwargs)


@pytest.mark.asyncio()
async def test_goal_success_custom_threshold():
    mock_llm = _make_mock_llm("3.5\nPartially achieved the goal.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm, score_threshold=3.0)

    result = await evaluator.aevaluate(
        query="Summarize this document",
        response="Here is a brief summary.",
    )
    assert result.score == 3.5
    assert result.passing is True  # 3.5 >= 3.0


@pytest.mark.asyncio()
async def test_goal_success_no_query_raises():
    mock_llm = _make_mock_llm("5.0\nPerfect.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    with pytest.raises(ValueError, match="query and response must be provided"):
        await evaluator.aevaluate(query=None, response="some response")


@pytest.mark.asyncio()
async def test_goal_success_no_response_raises():
    mock_llm = _make_mock_llm("5.0\nPerfect.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    with pytest.raises(ValueError, match="query and response must be provided"):
        await evaluator.aevaluate(query="some query", response=None)


@pytest.mark.asyncio()
async def test_goal_success_invalid_score():
    mock_llm = _make_mock_llm("not_a_number\nSomething went wrong.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    result = await evaluator.aevaluate(
        query="Do something",
        response="Done.",
    )
    assert result.score is None
    assert result.passing is None


@pytest.mark.asyncio()
async def test_goal_success_prompts():
    mock_llm = _make_mock_llm("4.0\nGood.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    prompts = evaluator.get_prompts()
    assert "eval_template" in prompts


@pytest.mark.asyncio()
async def test_goal_success_string_template():
    template = "Score this: {query} {generated_answer} {reference_answer} {tool_history}"
    mock_llm = _make_mock_llm("4.0\nGood.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm, eval_template=template)

    result = await evaluator.aevaluate(
        query="test query",
        response="test response",
    )
    assert result.score == 4.0


@pytest.mark.asyncio()
async def test_goal_success_result_fields():
    mock_llm = _make_mock_llm("4.5\nWell done.")
    evaluator = AgentGoalSuccessEvaluator(llm=mock_llm)

    result = await evaluator.aevaluate(
        query="Find flights",
        response="Found 3 flights to NYC.",
    )
    assert result.query == "Find flights"
    assert result.response == "Found 3 flights to NYC."
    assert result.score == 4.5
    assert result.passing is True
    assert result.feedback == "Well done."
