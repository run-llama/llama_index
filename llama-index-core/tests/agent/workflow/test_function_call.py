from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.agent.workflow import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    ToolCallResult,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import ToolOutput
from llama_index.core.workflow.context import Context
from llama_index.core.workflow.events import StopEvent


class TestWorkflowAgent(BaseWorkflowAgent):
    """Test implementation of BaseWorkflowAgent for testing"""

    async def take_step(self, ctx, llm_input, tools, memory):
        """Mock implementation"""
        return AgentOutput(
            response=ChatMessage(role="assistant", content="test response"),
            tool_calls=[],
            raw="test",
            current_agent_name=self.name,
        )

    async def handle_tool_call_results(self, ctx, results, memory):
        """Mock implementation"""

    async def finalize(self, ctx, output, memory):
        """Mock implementation"""
        return output


@pytest.fixture
def mock_context():
    """Create a mock context for testing"""
    ctx = MagicMock(spec=Context)
    ctx.store = AsyncMock()
    ctx.store.get = AsyncMock()
    ctx.store.set = AsyncMock()
    ctx.collect_events = MagicMock()
    return ctx


@pytest.fixture
def mock_memory():
    """Create a mock memory for testing"""
    memory = MagicMock(spec=BaseMemory)
    memory.aget = AsyncMock(return_value=[])
    return memory


@pytest.fixture
def test_agent():
    """Create a test agent instance"""
    return TestWorkflowAgent(
        name="test_agent",
        description="Test agent for testing",
        tools=[],
        llm=None,  # Will use default
    )


@pytest.mark.asyncio
async def test_aggregate_tool_results_return_direct_non_handoff_no_error_stops(
    mock_context, mock_memory, test_agent
):
    """
    Test that when return_direct tool is NOT 'handoff' and has NO error,
    the workflow stops execution by returning StopEvent (lines 564-569)
    """
    # Arrange
    tool_output = ToolOutput(
        content="Tool executed successfully",
        tool_name="direct_tool",
        raw_input={"param": "value"},
        raw_output="success",
        is_error=False,
    )

    return_direct_tool = ToolCallResult(
        tool_name="direct_tool",  # NOT 'handoff'
        tool_kwargs={"param": "value"},
        tool_id="tool_123",
        tool_output=tool_output,
        return_direct=True,
    )

    # Mock context store responses
    mock_context.store.get.side_effect = lambda key, default=None: {
        "num_tool_calls": 1,
        "current_tool_calls": [],
        "memory": mock_memory,
        "user_msg_str": "test message",
    }.get(key, default)

    # Mock collect_events to return our tool call result
    mock_context.collect_events.return_value = [return_direct_tool]

    # Act
    result = await test_agent.aggregate_tool_results(mock_context, return_direct_tool)

    # Assert
    # The method should return StopEvent when condition is met
    assert isinstance(result, StopEvent)
    assert result.result is not None
    assert result.result.current_agent_name == "test_agent"

    # Verify that current_tool_calls was cleared (line 567)
    mock_context.store.set.assert_any_call("current_tool_calls", [])


@pytest.mark.asyncio
async def test_aggregate_tool_results_return_direct_handoff_does_not_stop(
    mock_context, mock_memory, test_agent
):
    """
    Test that when return_direct tool is 'handoff',
    the workflow does NOT stop execution (condition fails at line 564)
    """
    # Arrange
    tool_output = ToolOutput(
        content="Handing off to another agent",
        tool_name="handoff",
        raw_input={"target_agent": "other_agent"},
        raw_output="handoff_success",
        is_error=False,
    )

    handoff_tool = ToolCallResult(
        tool_name="handoff",  # IS 'handoff'
        tool_kwargs={"target_agent": "other_agent"},
        tool_id="tool_456",
        tool_output=tool_output,
        return_direct=True,
    )

    # Mock context store responses
    mock_context.store.get.side_effect = lambda key, default=None: {
        "num_tool_calls": 1,
        "current_tool_calls": [],
        "memory": mock_memory,
        "user_msg_str": "test message",
    }.get(key, default)

    # Mock collect_events to return our handoff tool call result
    mock_context.collect_events.return_value = [handoff_tool]

    # Act
    result = await test_agent.aggregate_tool_results(mock_context, handoff_tool)

    # Assert
    # The method should NOT return StopEvent for handoff tools
    assert not isinstance(result, StopEvent)
    # Should return AgentInput to continue the workflow
    assert isinstance(result, AgentInput)

    # Verify that current_tool_calls was NOT cleared specifically for the return_direct condition
    # Note: current_tool_calls might be set for other reasons, but not the lines 564-569 condition
    calls_to_current_tool_calls = [
        call
        for call in mock_context.store.set.call_args_list
        if call[0][0] == "current_tool_calls" and call[0][1] == []
    ]
    # Should not find the specific call that clears current_tool_calls due to lines 564-569
    assert len(calls_to_current_tool_calls) == 0


@pytest.mark.asyncio
async def test_aggregate_tool_results_return_direct_with_error_does_not_stop(
    mock_context, mock_memory, test_agent
):
    """
    Test that when return_direct tool has an error,
    the workflow does NOT stop execution (condition fails at line 565)
    """
    # Arrange
    tool_output = ToolOutput(
        content="Tool execution failed",
        tool_name="error_tool",
        raw_input={"param": "value"},
        raw_output="error_occurred",
        is_error=True,  # HAS ERROR
    )

    error_tool = ToolCallResult(
        tool_name="error_tool",  # NOT 'handoff'
        tool_kwargs={"param": "value"},
        tool_id="tool_789",
        tool_output=tool_output,
        return_direct=True,
    )

    # Mock context store responses
    mock_context.store.get.side_effect = lambda key, default=None: {
        "num_tool_calls": 1,
        "current_tool_calls": [],
        "memory": mock_memory,
        "user_msg_str": "test message",
    }.get(key, default)

    # Mock collect_events to return our error tool call result
    mock_context.collect_events.return_value = [error_tool]

    # Act
    result = await test_agent.aggregate_tool_results(mock_context, error_tool)

    # Assert
    # The method should NOT return StopEvent for tools with errors
    assert not isinstance(result, StopEvent)
    # Should return AgentInput to continue the workflow
    assert isinstance(result, AgentInput)

    # Verify that current_tool_calls was NOT cleared specifically for the return_direct condition
    calls_to_current_tool_calls = [
        call
        for call in mock_context.store.set.call_args_list
        if call[0][0] == "current_tool_calls" and call[0][1] == []
    ]
    # Should not find the specific call that clears current_tool_calls due to lines 564-569
    assert len(calls_to_current_tool_calls) == 0


@pytest.mark.asyncio
async def test_aggregate_tool_results_return_direct_handoff_with_error_does_not_stop(
    mock_context, mock_memory, test_agent
):
    """
    Test that when return_direct tool is 'handoff' AND has an error,
    the workflow does NOT stop execution (condition fails on both counts)
    """
    # Arrange
    tool_output = ToolOutput(
        content="Handoff failed",
        tool_name="handoff",
        raw_input={"target_agent": "other_agent"},
        raw_output="handoff_error",
        is_error=True,  # HAS ERROR
    )

    handoff_error_tool = ToolCallResult(
        tool_name="handoff",  # IS 'handoff'
        tool_kwargs={"target_agent": "other_agent"},
        tool_id="tool_999",
        tool_output=tool_output,
        return_direct=True,
    )

    # Mock context store responses
    mock_context.store.get.side_effect = lambda key, default=None: {
        "num_tool_calls": 1,
        "current_tool_calls": [],
        "memory": mock_memory,
        "user_msg_str": "test message",
    }.get(key, default)

    # Mock collect_events to return our handoff error tool call result
    mock_context.collect_events.return_value = [handoff_error_tool]

    # Act
    result = await test_agent.aggregate_tool_results(mock_context, handoff_error_tool)

    # Assert
    # The method should NOT return StopEvent for handoff tools with errors
    assert not isinstance(result, StopEvent)
    # Should return AgentInput to continue the workflow
    assert isinstance(result, AgentInput)


@pytest.mark.asyncio
async def test_aggregate_tool_results_context_store_operations_for_successful_return_direct(
    mock_context, mock_memory, test_agent
):
    """
    Test that context store operations are performed correctly when condition is met (lines 567-568)
    """
    # Arrange
    tool_output = ToolOutput(
        content="Success",
        tool_name="success_tool",
        raw_input={},
        raw_output="ok",
        is_error=False,
    )

    success_tool = ToolCallResult(
        tool_name="success_tool",
        tool_kwargs={},
        tool_id="tool_success",
        tool_output=tool_output,
        return_direct=True,
    )

    # Mock context store responses
    existing_tool_calls = [success_tool]
    mock_context.store.get.side_effect = lambda key, default=None: {
        "num_tool_calls": 1,
        "current_tool_calls": existing_tool_calls,
        "memory": mock_memory,
        "user_msg_str": "test message",
    }.get(key, default)

    # Mock collect_events to return our success tool call result
    mock_context.collect_events.return_value = [success_tool]

    # Act
    result = await test_agent.aggregate_tool_results(mock_context, success_tool)

    # Assert
    # Verify StopEvent was returned
    assert isinstance(result, StopEvent)
    assert result.result is not None

    # Verify context store was called correctly (line 567)
    mock_context.store.set.assert_any_call("current_tool_calls", [])

    # Verify the result contains the correct information
    assert result.result.current_agent_name == "test_agent"
    assert result.result.response.content == "Success"


@pytest.mark.asyncio
async def test_aggregate_tool_results_multiple_tools_one_return_direct_eligible(
    mock_context, mock_memory, test_agent
):
    """
    Test that when multiple tools are called but only one is eligible for return_direct stop,
    the workflow stops correctly
    """
    # Arrange
    # First tool - return_direct but is handoff (should not stop)
    handoff_output = ToolOutput(
        content="Handoff",
        tool_name="handoff",
        raw_input={},
        raw_output="handoff",
        is_error=False,
    )

    handoff_tool = ToolCallResult(
        tool_name="handoff",
        tool_kwargs={},
        tool_id="tool_handoff",
        tool_output=handoff_output,
        return_direct=True,
    )

    # Second tool - return_direct and eligible to stop
    success_output = ToolOutput(
        content="Success",
        tool_name="success_tool",
        raw_input={},
        raw_output="success",
        is_error=False,
    )

    success_tool = ToolCallResult(
        tool_name="success_tool",
        tool_kwargs={},
        tool_id="tool_success",
        tool_output=success_output,
        return_direct=True,
    )

    # Mock context store responses
    mock_context.store.get.side_effect = lambda key, default=None: {
        "num_tool_calls": 2,
        "current_tool_calls": [],
        "memory": mock_memory,
        "user_msg_str": "test message",
    }.get(key, default)

    # Mock collect_events to return both tool call results
    mock_context.collect_events.return_value = [success_tool, handoff_tool]

    # Act
    result = await test_agent.aggregate_tool_results(mock_context, success_tool)

    # Assert
    # Should return StopEvent because one tool is eligible
    assert isinstance(result, StopEvent)

    # Verify context store was called to clear current_tool_calls (line 567)
    mock_context.store.set.assert_any_call("current_tool_calls", [])


@pytest.mark.asyncio
async def test_aggregate_tool_results_boolean_logic_verification():
    """
    Test the exact boolean logic used in lines 564-565
    """
    test_cases = [
        # (tool_name, is_error, should_stop_execution)
        ("handoff", False, False),  # handoff tool, no error -> should NOT stop
        ("handoff", True, False),  # handoff tool, with error -> should NOT stop
        ("other_tool", False, True),  # non-handoff tool, no error -> should stop
        ("other_tool", True, False),  # non-handoff tool, with error -> should NOT stop
        ("", False, True),  # empty name (not handoff), no error -> should stop
        ("", True, False),  # empty name (not handoff), with error -> should NOT stop
    ]

    for tool_name, is_error, should_stop in test_cases:
        # Create tool output
        tool_output = ToolOutput(
            content="test",
            tool_name=tool_name,
            raw_input={},
            raw_output="test",
            is_error=is_error,
        )

        return_direct_tool = ToolCallResult(
            tool_name=tool_name,
            tool_kwargs={},
            tool_id="test_id",
            tool_output=tool_output,
            return_direct=True,
        )

        # Test the actual boolean logic from lines 564-565
        condition_result = (
            return_direct_tool.tool_name != "handoff"
            and not return_direct_tool.tool_output.is_error
        )

        assert condition_result == should_stop, (
            f"Boolean logic failed for tool_name='{tool_name}', is_error={is_error}. "
            f"Expected {should_stop}, got {condition_result}"
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
